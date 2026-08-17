"""
ScadaTCNEncoder:
in: x    (B, C, T) float32   raw sensor value
    mask (B, C, T) float32   1 = observed, 0 = missing
out:     (B, C * P + n_side, d) float32   tokens

PerceiverResampler:
in: (B, S, d) -> out: (B, n_latents, d)    e.g. (4, 32, 128)

B = Batch - how many windows you process at once
C = Channels - how many sensors
T = time step - how many time steps in each batch
P = tokens per channel
d = token width
C * P = total number of tokens (channels * tokens per channel)
S = flat sequence length C * P + n_side
n_side = number of side tokens - 3 categorical tokens (turbine state, alarm code, curtailment flag), 1 static token (the MLP over day-of-year, hour, operating hours, cycles, days since maintenance)
n_latents = resampler output tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Stem(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.proj = nn.Conv1d(2, d_model, kernel_size=1)

    def forward(self, x, mask):
        # x:    (N, T)
        # mask: (N, T)
        h = torch.stack([x, mask], 1)
        h = self.proj(h)
        
        return h

class CausalConv(nn.Module):
    def __init__(self, d_model=128, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size-1) * dilation
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, dilation=dilation)

    def forward(self, h):

        h = F.pad(h, (self.pad, 0))
        h = self.conv(h)

        return h


class ChannelNorm(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h):
        h = h.transpose(1, 2)
        h = self.norm(h)
        h = h.transpose(1, 2)

        return h


class CausalBlock(nn.Module):
    def __init__(self, d_model=128, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv(d_model=d_model, kernel_size=kernel_size, dilation=dilation)
        self.norm1 = ChannelNorm(d_model=d_model)
        self.conv2 = CausalConv(d_model=d_model, kernel_size=kernel_size, dilation=dilation)
        self.norm2 = ChannelNorm(d_model=d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        r = h
        h = self.conv1(h)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.drop(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.gelu(h)
        h = self.drop(h)

        return h + r

class TCNTrunk(nn.Module):
    def __init__(self, d_model=128, kernel_size=3, dilations=(1, 2, 4, 8, 16, 32, 64, 128), dropout=0.1):
        super().__init__()
        self.stem = Stem(d_model)
        self.blocks = nn.ModuleList([CausalBlock(d_model, kernel_size, d, dropout) for d in dilations])
        self.receptive_field = 1 + 2 * sum((kernel_size - 1) * d for d in dilations)

    def forward(self, x, mask):
        h = self.stem(x, mask)
        for blk in self.blocks:
            h = blk(h)

        return h

class RevIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, mask):
        n = mask.sum(-1, keepdim=True).clamp(min=1.0)
        mean = (x * mask).sum(-1, keepdim=True) / n
        var = (((x - mean) * mask) ** 2).sum(-1, keepdim=True) / n
        std = (var + self.eps).sqrt()

        x = (x - mean) / std

        return x * mask

class ScadaTCNEncoder(nn.Module):
    def __init__(self, d_model=128,
                 kernel_size=3,
                 dilations=(1, 2, 4, 8, 16, 32, 64, 128),
                 dropout=0.1,
                 stride=15,
                 context_len=600,
                 n_channels=20,
                 categorical_cardinalities=(12, 256, 2),
                 n_static=7):
        super().__init__()    
        self.d_model = d_model
        self.revin = RevIN()
        self.trunk = TCNTrunk(d_model=d_model, kernel_size=kernel_size, dilations=dilations, dropout=dropout)
        self.stride = stride
        self.out_len = len(range(self.stride - 1, context_len, stride))
        self.channel_embed = nn.Embedding(n_channels, d_model)
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(card, d_model) for card in categorical_cardinalities
        ])
        self.static_proj = nn.Sequential(
            nn.Linear(n_static, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        assert self.trunk.receptive_field >= context_len, \
        f"receptive field {self.trunk.receptive_field} < context {context_len}"

    def forward(self, x, mask, categorical=None, static_feats=None):
        B, C, T = x.shape
        xf = x.reshape(B * C, T)
        mf = mask.reshape(B * C, T)

        h = self.revin(xf, mf)
        h = self.trunk(h, mf)

        h = h.reshape(B, C, self.d_model, T)

        h = h[:, :, :, self.stride-1::self.stride]
        h = h.transpose(2, 3)

        ch = torch.arange(C, device=x.device)
        h = h + self.channel_embed(ch)[None, :, None, :]

        seq = h.reshape(B, C * self.out_len, self.d_model)

        if categorical is not None:
            cat_tokens = torch.stack(
                [emb(categorical[:, i]) for i, emb in enumerate(self.cat_embeds)], dim=1
            )
            seq = torch.cat([seq, cat_tokens], dim=1)

        if static_feats is not None:
            static_token = self.static_proj(static_feats).unsqueeze(1)
            seq = torch.cat([seq, static_token], dim=1)

        return seq


# ResamplerLayer and PerceiverResampler used to live here. They are SHARED --
# every modality collapses its own token count onto the same 32 latents with
# them -- so they now live in models/common.py and are re-exported here only so
# older imports keep working.
#
# Import them from models.common in new code. Two copies is how they drifted
# apart in the first place: this file's version was missing the
# need_weights=False that common.py's has, and since every caller imported from
# HERE, the fix was dead code and every run paid ~37% extra in the resampler.
from models.common import ResamplerLayer, PerceiverResampler  # noqa: F401,E402
