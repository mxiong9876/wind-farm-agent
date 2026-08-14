import torch
import torch.nn as nn
import math

class ContinuousTimeEncoding(nn.Module):
    """
    ContinuousTimeEncoding:
    in : t (B, P) float32   seconds relative to the window's right edge
    out:   (B, P, d) float32

    CONVENTION -- every encoder must match this or cross-modal alignment breaks:
        t = 0     at the window's right edge ("now")
        units     seconds
        sign      negative going into the past, so -180.0 is three minutes ago

    n_freqs    = number of sine/cosine pairs; 32 pairs -> 64 features -> d_model
    min_period = fastest oscillation in seconds (1 s). The finest time difference
                the encoding can distinguish.
    max_period = slowest oscillation in seconds (604800 = one week). The longest
                span before the encoding repeats and two times become confusable.

    The frequencies are log-spaced between the two, so seconds, minutes, hours and
    days all get comparable resolution. They are register_buffer, not Parameter:
    fixed by design, but they still have to follow the model to GPU and survive
    a check point round trip.
    """

    def __init__(self, d_model=128, n_freqs=32, min_period=1.0, max_period=604800.0):
        super().__init__()
        periods = torch.logspace(math.log10(min_period),
                                 math.log10(max_period), n_freqs)
        self.register_buffer("omega", 2 * math.pi / periods)
        self.proj = nn.Linear(2 * n_freqs, d_model)

    def forward(self, t):
        # t: (B, P) seconds relative to now
        phase = t.unsqueeze(-1) * self.omega       # (B, P, n_freqs)
        feats = torch.cat([phase.sin(), phase.cos()], dim=-1)
        return self.proj(feats)    


class ResamplerLayer(nn.Module):
    def __init__(self, d_model=128, n_heads=8):
        super().__init__()
        self.norm_l  = nn.LayerNorm(d_model)
        self.norm_x  = nn.LayerNorm(d_model)
        self.attn    = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff      = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))

    def forward(self, z, x):
        q = self.norm_l(z)
        kv = self.norm_x(x)
        z = z + self.attn(q, kv, kv, need_weights=False)[0]
        z = z + self.ff(self.norm_ff(z))

        return z

class PerceiverResampler(nn.Module):
    def __init__(self, d_model=128, n_latents=32, n_heads=8, n_layers=2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.layers = nn.ModuleList([ResamplerLayer(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, x):
        z = self.latents.unsqueeze(0).expand(x.size(0), -1, -1)
        for layer in self.layers:
            z = layer(z, x)

        return z