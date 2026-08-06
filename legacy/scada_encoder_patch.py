"""
SCADA encoder head for multimodal wind turbine health monitoring.

Contract:
    input : continuous window (B, C, T), observed mask (B, C, T), categorical (B, K)
    output: (B, N_latents, d_model) tokens for the fusion module

Design: RevIN -> channel-independent patching -> shared transformer
        -> channel identity re-injection -> Perceiver resampler.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """Reversible instance normalization, computed over observed values only.

    Normalizes each (batch, channel) series by its own statistics so the encoder
    generalizes across turbine models, seasons, and operating regimes.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, mask):
        # x, mask: (B, C, T). mask is 1.0 where observed.
        n = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = (x * mask).sum(dim=-1, keepdim=True) / n
        var = (((x - mean) * mask) ** 2).sum(dim=-1, keepdim=True) / n
        std = (var + self.eps).sqrt()

        x = (x - mean) / std
        x = x * mask  # zero out unobserved positions after centering

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        self._stats = (mean, std)
        return x

    def denormalize(self, x):
        mean, std = self._stats
        if self.affine:
            x = (x - self.bias[None, :, None]) / self.weight[None, :, None]
        return x * std + mean


class Patchify(nn.Module):
    """Split (B, C, T) into overlapping patches -> (B, C, num_patches, patch_len)."""

    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def num_patches(self, seq_len: int) -> int:
        return (seq_len - self.patch_len) // self.stride + 1

    def forward(self, x):
        # pad the tail so no trailing samples are silently dropped
        seq_len = x.size(-1)
        n = self.num_patches(seq_len)
        needed = (n - 1) * self.stride + self.patch_len
        if needed < seq_len:
            pad = self.stride - (seq_len - needed) % self.stride
            x = F.pad(x, (0, pad), mode="replicate")
        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)


class ContinuousTimeEncoding(nn.Module):
    """Fourier features of elapsed time relative to the window's right edge.

    Unlike an index-based positional embedding, this is comparable ACROSS
    modalities: t = -600.0 means "600 seconds ago" whether it came from SCADA
    at 1 Hz or from vibration at 25 kHz. This is what lets the fusion module
    align patches that actually co-occur in wall-clock time.
    """

    def __init__(self, d_model: int, n_freqs: int = 32,
                 min_period: float = 1.0, max_period: float = 604800.0):
        super().__init__()
        # log-spaced periods from 1 second out to one week
        periods = torch.logspace(
            math.log10(min_period), math.log10(max_period), n_freqs
        )
        self.register_buffer("omega", 2 * math.pi / periods)
        self.proj = nn.Linear(2 * n_freqs, d_model)

    def forward(self, t):
        # t: (B, P) seconds relative to "now" (negative = past)
        phase = t.unsqueeze(-1) * self.omega  # (B, P, n_freqs)
        feats = torch.cat([phase.sin(), phase.cos()], dim=-1)
        return self.proj(feats)  # (B, P, d_model)


def patch_center_times(context_len: int, patch_len: int, stride: int,
                       sample_period: float, device=None):
    """Center timestamp of each patch, in seconds before the window end."""
    n = (context_len - patch_len) // stride + 1
    idx = torch.arange(n, device=device, dtype=torch.float32)
    centers = idx * stride + patch_len / 2.0
    return -(context_len - centers) * sample_period  # (P,), negative = past


class PerceiverResampler(nn.Module):
    """Compress a variable-length token sequence to N fixed learned latents."""

    def __init__(self, d_model: int, n_latents: int = 32, n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict({
                    "norm_l": nn.LayerNorm(d_model),
                    "norm_x": nn.LayerNorm(d_model),
                    "attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                    "norm_ff": nn.LayerNorm(d_model),
                    "ff": nn.Sequential(
                        nn.Linear(d_model, 4 * d_model), nn.GELU(),
                        nn.Linear(4 * d_model, d_model),
                    ),
                })
            )

    def forward(self, x, key_padding_mask=None):
        # x: (B, S, d_model) -> (B, n_latents, d_model)
        z = self.latents.unsqueeze(0).expand(x.size(0), -1, -1)
        for layer in self.layers:
            q = layer["norm_l"](z)
            kv = layer["norm_x"](x)
            attn_out, _ = layer["attn"](q, kv, kv, key_padding_mask=key_padding_mask)
            z = z + attn_out
            z = z + layer["ff"](layer["norm_ff"](z))
        return z


class ScadaEncoder(nn.Module):
    def __init__(
        self,
        n_continuous: int,
        categorical_cardinalities: tuple = (),
        context_len: int = 600,
        patch_len: int = 30,
        stride: int = 15,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        n_latents: int = 32,
        n_turbine_types: int = 8,
        n_static_feats: int = 0,
        sample_period: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_continuous = n_continuous
        self.d_model = d_model
        self.context_len = context_len
        self.patch_len = patch_len
        self.stride = stride
        self.sample_period = sample_period

        self.revin = RevIN(n_continuous)
        self.patchify = Patchify(patch_len, stride)
        n_patches = self.patchify.num_patches(context_len)

        # patch_len values + patch_len mask entries, shared across all channels
        self.patch_proj = nn.Linear(patch_len * 2, d_model)
        self.pos_embed = nn.Parameter(torch.randn(n_patches, d_model) * 0.02)
        self.channel_embed = nn.Embedding(n_continuous, d_model)
        self.turbine_embed = nn.Embedding(n_turbine_types, d_model)

        # cross-modality temporal alignment (see ContinuousTimeEncoding docstring)
        self.time_encoding = ContinuousTimeEncoding(d_model)

        # modality identity, added at the encoder OUTPUT so fusion can tell
        # SCADA tokens apart from vibration/acoustic/RGBD even if a modality drops
        self.modality_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        # per-window scalars that no window can recover: day-of-year sin/cos,
        # hour-of-day sin/cos, operating hours, cycles, time since maintenance
        self.static_proj = nn.Sequential(
            nn.Linear(n_static_feats, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        ) if n_static_feats > 0 else None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.cat_embeds = nn.ModuleList(
            [nn.Embedding(card, d_model) for card in categorical_cardinalities]
        )

        self.resampler = PerceiverResampler(d_model, n_latents=n_latents, n_heads=n_heads)

        # pretraining heads (drop or freeze at fusion time)
        self.recon_head = nn.Linear(d_model, patch_len)
        self.power_curve_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def encode_patches(self, x, mask, turbine_type=None, patch_times=None):
        """x, mask: (B, C, T) -> per-channel patch tokens (B, C, P, d_model).

        patch_times: (B, P) seconds relative to window end. If omitted, derived
        from the configured sampling period assuming a regular grid.
        """
        B, C, _ = x.shape
        x = self.revin(x, mask)

        xp = self.patchify(x)      # (B, C, P, patch_len)
        mp = self.patchify(mask)   # (B, C, P, patch_len)
        P = xp.size(2)

        tokens = self.patch_proj(torch.cat([xp, mp], dim=-1))  # (B, C, P, d)

        # intra-modality order
        tokens = tokens + self.pos_embed[None, None, :P, :]

        # cross-modality wall-clock alignment
        if patch_times is None:
            patch_times = patch_center_times(
                self.context_len, self.patch_len, self.stride,
                self.sample_period, device=x.device,
            ).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.time_encoding(patch_times)[:, None, :, :]

        # channel-independent: fold channels into the batch dimension
        tokens = tokens.reshape(B * C, P, self.d_model)
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens).reshape(B, C, P, self.d_model)

        # re-inject channel identity so fusion knows which sensor each token came from
        ch_ids = torch.arange(C, device=x.device)
        tokens = tokens + self.channel_embed(ch_ids)[None, :, None, :]

        if turbine_type is not None:
            tokens = tokens + self.turbine_embed(turbine_type)[:, None, None, :]

        return tokens

    def forward(self, x, mask, categorical=None, turbine_type=None,
                patch_times=None, static_feats=None):
        """Returns (B, n_latents, d_model) tokens for the fusion module."""
        B = x.size(0)
        tokens = self.encode_patches(x, mask, turbine_type, patch_times)
        seq = tokens.reshape(B, -1, self.d_model)  # (B, C*P, d)

        if categorical is not None and len(self.cat_embeds) > 0:
            cat_tokens = torch.stack(
                [emb(categorical[:, i]) for i, emb in enumerate(self.cat_embeds)], dim=1
            )
            seq = torch.cat([seq, cat_tokens], dim=1)

        if static_feats is not None and self.static_proj is not None:
            seq = torch.cat([seq, self.static_proj(static_feats).unsqueeze(1)], dim=1)

        out = self.resampler(seq)
        return out + self.modality_embed  # tag as SCADA for the fusion module

    # ------------------------------------------------------------------
    # self-supervised pretraining
    # ------------------------------------------------------------------
    def masked_reconstruction_loss(self, x, mask, mask_ratio: float = 0.4):
        """Mask whole patches and reconstruct them in normalized space."""
        B, C, _ = x.shape
        x_norm = self.revin(x, mask)
        target = self.patchify(x_norm)          # (B, C, P, patch_len)
        target_mask = self.patchify(mask)
        P = target.size(2)

        keep = (torch.rand(B, C, P, 1, device=x.device) > mask_ratio).float()
        corrupted = self.patch_proj(
            torch.cat([target * keep, target_mask * keep], dim=-1)
        )
        corrupted = corrupted + self.pos_embed[None, None, :P, :]
        h = self.transformer(corrupted.reshape(B * C, P, self.d_model))
        pred = self.recon_head(self.norm(h)).reshape(B, C, P, -1)

        loss_mask = (1.0 - keep) * target_mask
        return ((pred - target) ** 2 * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

    def power_curve_loss(self, tokens, power_target, wind_channel_idx: int):
        """Auxiliary physics head: predict normalized power from the wind-speed
        channel's tokens. The residual doubles as a degradation indicator."""
        wind_tokens = tokens[:, wind_channel_idx].mean(dim=1)  # (B, d)
        pred = self.power_curve_head(wind_tokens).squeeze(-1)
        return F.mse_loss(pred, power_target)


if __name__ == "__main__":
    B, C, T = 4, 40, 600
    enc = ScadaEncoder(
        n_continuous=C,
        categorical_cardinalities=(12, 256, 2),
        n_static_feats=7,     # doy sin/cos, hour sin/cos, op hours, cycles, t-since-maint
        sample_period=1.0,    # 1 Hz logging
    )

    x = torch.randn(B, C, T)
    mask = (torch.rand(B, C, T) > 0.05).float()
    cat = torch.stack([
        torch.randint(0, 12, (B,)),
        torch.randint(0, 256, (B,)),
        torch.randint(0, 2, (B,)),
    ], dim=1)

    out = enc(
        x, mask, cat,
        turbine_type=torch.zeros(B, dtype=torch.long),
        static_feats=torch.randn(B, 7),
    )
    print("fusion tokens:", out.shape)
    print("patch times (s before now):",
          patch_center_times(600, 30, 15, 1.0)[:5].tolist(), "...")
    print("recon loss:", enc.masked_reconstruction_loss(x, mask).item())
    print("params (M):", sum(p.numel() for p in enc.parameters()) / 1e6)