"""Stage 6 -- RevIN: per-window, per-channel normalization over observed values.

RevIN is intentionally NON-causal: it uses whole-window statistics. That is
correct for the health path. For the control path it must be replaced with a
running-statistics variant, since future samples do not exist at decision time.
"""

import torch
from scada_encoder_tcn import RevIN
from helpers import banner, check_shape, report


def test():
    banner("stage 6: revin")
    torch.manual_seed(0)
    N, T = 80, 600
    x = torch.randn(N, T) * 4.0 + 30.0        # gearbox temp in January
    m = (torch.rand(N, T) > 0.05).float()

    rev = RevIN()
    xn = rev(x, m)
    check_shape("shape preserved", xn, (N, T))

    mean = xn.mean(-1)[0].item()
    std = xn.std(-1)[0].item()
    report("mean over time ~ 0", abs(mean), abs(mean) < 0.02, fmt="{:.4f}")
    report("std over time ~ 1", abs(std - 1.0), abs(std - 1.0) < 0.05,
           fmt="{:.4f}")

    # the whole point: a seasonal offset must vanish
    shift = (rev(x, m) - rev(x + 50.0, m)).abs().max().item()
    report("invariant to +50 offset (summer)", shift, shift < 1e-3)

    # and a scale change (different turbine model) must vanish too
    scale = (rev(x, m) - rev(x * 3.0, m)).abs().max().item()
    report("invariant to 3x scale", scale, scale < 1e-3)

    # statistics must ignore masked positions, otherwise a half-missing channel
    # gets a mean dragged halfway to zero
    xg = x.clone()
    mg = m.clone()
    mg[:, 300:] = 0.0
    xg[:, 300:] = 0.0                          # missing entries stored as 0
    out = rev(xg, mg)
    obs_mean = (out * mg).sum(-1) / mg.sum(-1).clamp(min=1)
    err = obs_mean.abs().max().item()
    report("mask excluded from statistics", err, err < 0.02, fmt="{:.4f}")

    # masked positions must come out exactly 0, not -mean/std
    leftover = (out * (1 - mg)).abs().max().item()
    report("masked positions re-zeroed", leftover, leftover < 1e-6)

    # an all-missing channel must not divide by zero
    dead = rev(torch.zeros(1, T), torch.zeros(1, T))
    finite = bool(torch.isfinite(dead).all())
    report("all-missing channel is finite", float(finite), finite, fmt="{:.0f}")


if __name__ == "__main__":
    test()
