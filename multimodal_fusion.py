"""
MultiModalFusion:
in : inputs  {name: {kwarg: tensor}}   per-modality encoder arguments
     present (B, M) float32            1 = turbine has this modality, 0 = it does not
out:         (B, d) float32            one health state per turbine

B = Batch      - how many turbine-windows you process at once
M = Modalities - how many sensor families are registered (SCADA, vibration, ...)
K = n_latents  - latents each modality is compressed to (32)
d = token width (128, matching both encoders)

WHY A REGISTRY AND NOT A TWO-ARGUMENT FUNCTION

Real fleets are ragged. One turbine has a CMS, the next does not; a blade
camera is retrofitted to a third; an acoustic array shows up on the ten newest
machines only. A fusion model hard-wired to "SCADA and vibration" cannot
represent that, and generalising it later means rewriting the attention mask,
the pooling and the training loop at once. So modalities are registered by
name and every one of them is optional at every step.

THE INTERFACE THAT MAKES THAT POSSIBLE

Encoders emit wildly different token counts -- SCADA 804, vibration 48, a
future acoustic encoder something else again. Each modality therefore gets its
OWN PerceiverResampler collapsing its tokens to a fixed K=32. After that point
every modality occupies exactly K slots of width d and the fusion body cannot
tell them apart except through a learned modality embedding, the same way
channel_embed tells each encoder which sensor a token came from.

That fixed width is not just tidiness. Concatenating raw tokens into one shared
resampler would make SCADA 804 of 852 keys -- 94% -- and a resampler at
initialisation behaves close to uniform averaging, so vibration would start at
roughly a 6% contribution and have to climb out. Per-modality resamplers remove
the imbalance by construction: 32 keys each, whatever they started with.

K is uniform across modalities on purpose. SCADA carries more raw information
than vibration, so it is tempting to give it more latents -- but unequal blocks
make presence masking, dropout and pooling all special-case per modality, and
attention can learn to weight SCADA higher anyway. The wiring stays symmetric;
the model decides what matters.

TWO KINDS OF MISSING, KEPT SEPARATE

  intra-modality  a dead accelerometer on a turbine that HAS a CMS.
                  Already handled inside each encoder by its own channel mask.
  inter-modality  a turbine with no CMS at all.
                  Handled here, by `present`, which gates whole K-blocks out
                  of the attention.

Conflating them means feeding an encoder all-zeros and hoping the tokens come
out neutral. They do not: a silent snapshot is a perfectly valid input that
means "this sensor reads zero", which is not the same claim as "this sensor
does not exist".

    from multimodal_fusion import MultiModalFusion, StubEncoder
    fusion = MultiModalFusion({"scada": scada_enc, "vibration": vib_enc,
                               "acoustic": StubEncoder(), "rgbd": StubEncoder()})
    health = fusion({"scada": {"x": xs, "mask": ms},
                     "vibration": {"x": xv, "mask": mv}})     # (B, 128)
"""

import torch
import torch.nn as nn

from scada_encoder_tcn import PerceiverResampler


class StubEncoder(nn.Module):
    """Placeholder for a modality whose encoder is not written yet.

    Exists so the 4-modality routing, presence masking and dropout paths can be
    exercised TODAY, with acoustic and RGB-D standing in. Most of the code in
    this file is routing, and routing bugs -- an off-by-one into the modality
    embedding, a mask that assumes exactly two blocks, a pool that divides by
    the wrong count -- only surface once there are enough modalities to route.
    Finding them now beats finding them while also debugging a new encoder.

    It reads its input rather than returning a constant, so gradient-flow tests
    over the whole graph stay meaningful. It makes no claim to be a good model
    of anything.
    """

    def __init__(self, d_model=128, n_tokens=64, n_channels=4):
        super().__init__()
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.n_channels = n_channels
        self.proj = nn.Linear(n_channels, d_model)
        self.pos = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)

    def forward(self, x, mask=None):
        # x: (B, C, L) -> (B, n_tokens, d_model)
        B, C, L = x.shape
        assert C == self.n_channels, \
            f"got {C} channels, built for {self.n_channels}"
        assert L >= self.n_tokens, \
            f"{L} samples cannot fill {self.n_tokens} tokens"
        if mask is not None:
            x = x * mask[..., None]
        keep = L - L % self.n_tokens
        chunks = x[..., :keep].reshape(B, C, self.n_tokens, -1)
        return self.proj(chunks.mean(-1).transpose(1, 2)) + self.pos


class FusionBlock(nn.Module):
    """Pre-norm self-attention over the concatenated latents of every modality.

    Deliberately modality-agnostic: a flat sequence plus embeddings, with no
    pairwise cross-attention and no per-modality branches. Whether SCADA
    matters more than RGB-D is a fact about the data, so it belongs in learned
    attention weights, not in the wiring.
    """

    def __init__(self, d_model=128, n_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, h, pad_mask):
        q = self.norm1(h)
        a = self.attn(q, q, q, key_padding_mask=pad_mask, need_weights=False)[0]
        h = h + self.drop(a)
        return h + self.ff(self.norm2(h))


class MaskedAttentionPool(nn.Module):
    """Pool present latents to one vector with a learned query.

    A mean over present blocks would work, but its statistics move with how
    many modalities showed up, and the head downstream then sees a different
    input distribution for a 2-modality turbine than for a 4-modality one.
    Attention weights sum to one over the PRESENT keys, so the pooled scale is
    the same either way.
    """

    def __init__(self, d_model=128, n_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, h, pad_mask):
        kv = self.norm_kv(h)
        q = self.query.expand(h.size(0), -1, -1)
        out = self.attn(q, kv, kv, key_padding_mask=pad_mask,
                        need_weights=False)[0]
        return self.norm_out(out.squeeze(1))


class MultiModalFusion(nn.Module):
    """N optional modalities -> per-modality latents -> one health state."""

    def __init__(self, encoders, d_model=128, n_latents=32, n_heads=8,
                 n_layers=2, resampler_layers=2, dropout=0.1,
                 modality_dropout=0.15):
        super().__init__()
        assert len(encoders) > 0, "fusion needs at least one modality"
        assert 0.0 <= modality_dropout < 1.0, \
            f"modality_dropout {modality_dropout} must be in [0, 1)"

        self.names = list(encoders.keys())
        self.d_model = d_model
        self.n_latents = n_latents
        self.modality_dropout = modality_dropout

        self.encoders = nn.ModuleDict(encoders)
        # one resampler per modality -- this is what makes the blocks equal-sized
        self.resamplers = nn.ModuleDict({
            n: PerceiverResampler(d_model, n_latents, n_heads, resampler_layers)
            for n in self.names
        })
        # normalise each modality's block BEFORE fusion, so one encoder's output
        # scale drifting during training cannot quietly swamp the others
        self.latent_norms = nn.ModuleDict({
            n: nn.LayerNorm(d_model) for n in self.names
        })

        self.modality_embed = nn.Embedding(len(self.names), d_model)
        # A finite stand-in for an absent block, so the concatenated tensor is
        # always well defined. It is a BUFFER, not a Parameter, and that is not
        # an oversight: `pad` masks these exact positions out of every attention
        # that follows, so no gradient can ever reach them. Registering it as a
        # Parameter would put 512 permanently-dead weights in the optimizer and
        # imply to anyone reading this that absence is something the model
        # learns a representation for. It is not -- absence is handled by the
        # mask. The values are small and distinct per modality rather than zero
        # so that a masking bug shows up as a visible, finite perturbation
        # instead of silently reading as a neutral all-zero modality.
        #
        # Making absence INFORMATIVE -- letting the model condition on "this
        # turbine has no CMS, so a bearing fault cannot be ruled out" -- is a
        # real and defensible design, but it is a different one: it means not
        # padding these blocks at all, and it needs its own tests.
        self.register_buffer(
            "missing", torch.randn(len(self.names), d_model) * 0.02)

        self.blocks = nn.ModuleList([
            FusionBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.pool = MaskedAttentionPool(d_model, n_heads)

    # -- presence bookkeeping ------------------------------------------------
    def _batch_and_device(self, inputs):
        for name in self.names:
            for v in inputs.get(name, {}).values():
                if torch.is_tensor(v):
                    return v.shape[0], v.device
        raise ValueError("no tensor found in inputs; cannot infer batch size")

    def _presence(self, inputs, present, B, device):
        if present is None:
            flags = [1.0 if n in inputs else 0.0 for n in self.names]
            return torch.tensor(flags, device=device).expand(B, -1).clone()
        assert present.shape == (B, len(self.names)), \
            f"present {tuple(present.shape)} != {(B, len(self.names))}"
        # a modality with no input tensors cannot be present for anyone
        absent = torch.tensor([0.0 if n in inputs else 1.0 for n in self.names],
                              device=device)
        return present.to(device).float() * (1.0 - absent)

    def _drop_modalities(self, present):
        """Randomly hide modalities during training, never all of them.

        Without this, "runs without a couple of them" is true structurally and
        false in practice: trained only on complete turbines, the head collapses
        onto whichever stream is most informative and has no reason to build a
        backup path, so it falls apart the first time that stream is absent.
        """
        if not self.training or self.modality_dropout <= 0.0:
            return present
        keep = (torch.rand_like(present) >= self.modality_dropout).float()
        out = present * keep
        stranded = out.sum(dim=1) == 0
        if bool(stranded.any()):
            # hand each stranded sample back one modality it actually had
            idx = torch.multinomial(present[stranded], 1)
            out[stranded] = out[stranded].scatter(1, idx, 1.0)
        return out

    # -- forward -------------------------------------------------------------
    def forward(self, inputs, present=None, return_parts=False):
        B, device = self._batch_and_device(inputs)
        present = self._presence(inputs, present, B, device)
        assert bool((present.sum(dim=1) > 0).all()), \
            "every sample needs at least one present modality"
        eff = self._drop_modalities(present)

        blocks = []
        for m, name in enumerate(self.names):
            miss = self.missing[m].view(1, 1, -1)
            if name in inputs and bool(eff[:, m].max() > 0):
                z = self.resamplers[name](self.encoders[name](**inputs[name]))
                z = self.latent_norms[name](z)
            else:
                # absent for the WHOLE batch: skip the encoder entirely. The
                # SCADA trunk is seconds per step, so this is real time saved.
                z = miss.expand(B, self.n_latents, -1)
            on = eff[:, m].view(B, 1, 1)
            z = z * on + miss * (1.0 - on)
            blocks.append(z + self.modality_embed.weight[m])

        h = torch.cat(blocks, dim=1)                       # (B, M*K, d)
        pad = eff.repeat_interleave(self.n_latents, dim=1) == 0   # True = ignore
        for blk in self.blocks:
            h = blk(h, pad)
        health = self.pool(h, pad)

        if return_parts:
            return health, {"tokens": h, "present": eff, "pad_mask": pad}
        return health


if __name__ == "__main__":
    from scada_encoder_tcn import ScadaTCNEncoder
    from vibration_encoder_2dconv import VibrationConv2dEncoder

    torch.manual_seed(0)
    B = 2
    fusion = MultiModalFusion({
        "scada": ScadaTCNEncoder(n_channels=6, context_len=120,
                                 dilations=(1, 2, 4, 8, 16)),
        "vibration": VibrationConv2dEncoder(n_channels=8),
        "acoustic": StubEncoder(n_tokens=64, n_channels=4),
        "rgbd": StubEncoder(n_tokens=32, n_channels=3),
    }).eval()

    inputs = {
        "scada": {"x": torch.randn(B, 6, 120),
                  "mask": torch.ones(B, 6, 120)},
        "vibration": {"x": torch.randn(B, 8, 25600),
                      "mask": torch.ones(B, 8)},
        "acoustic": {"x": torch.randn(B, 4, 16000),
                     "mask": torch.ones(B, 4)},
        "rgbd": {"x": torch.randn(B, 3, 4096), "mask": torch.ones(B, 3)},
    }

    with torch.no_grad():
        health, parts = fusion(inputs, return_parts=True)
        # turbine 0 keeps everything, turbine 1 has SCADA only
        p = torch.tensor([[1., 1., 1., 1.], [1., 0., 0., 0.]])
        partial = fusion(inputs, present=p)

    n = sum(par.numel() for par in fusion.parameters())
    print(f"modalities     {fusion.names}")
    print(f"fused tokens   {tuple(parts['tokens'].shape)}"
          f"   = {len(fusion.names)} x {fusion.n_latents} latents")
    print(f"health state   {tuple(health.shape)}")
    print(f"partial ok     {tuple(partial.shape)}  "
          f"finite={bool(torch.isfinite(partial).all())}")
    print(f"parameters     {n:,}  ({n * 4 / 1e6:.1f} MB fp32)")
