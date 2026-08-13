"""Stage 3 -- ChannelNorm: normalize features per timestep, not across time.

This is the stage where BatchNorm/GroupNorm would silently destroy causality.
"""

import torch
from models.scada_encoder_tcn import ChannelNorm
from helpers import banner, check_shape, check_causal, report


def test():
    banner("stage 3: channel norm")
    torch.manual_seed(0)
    N, T, d = 80, 600, 128
    h = torch.randn(N, d, T) * 3.0 + 7.0     # off-centre and wrongly scaled

    norm = ChannelNorm(d_model=d)
    check_shape("shape preserved", norm(h), (N, d, T))
    check_causal(norm, label="causal (feature-axis norm)")

    norm.eval()
    with torch.no_grad():
        out = norm(h)

    # dim=1 is the FEATURE axis. Statistics must be ~0/~1 here.
    mean = out.mean(dim=1)[0, 0].item()
    std = out.std(dim=1)[0, 0].item()
    report("mean over features ~ 0", abs(mean), abs(mean) < 1e-4)
    report("std over features ~ 1", abs(std - 1.0), abs(std - 1.0) < 0.05,
           fmt="{:.4f}")

    # If the two transposes cancelled out, shape/causality/mean/std all still
    # pass while the TIME axis got normalized instead. Output statistics alone
    # cannot distinguish the two, so compare directly against the wrong version.
    with torch.no_grad():
        wrong = torch.nn.LayerNorm(T)(h)          # normalizes time, not features
    gap = (out - wrong).abs().max().item()
    report("not normalizing the time axis", gap, gap > 0.1, fmt="{:.4f}")

    # eps must be small; nn.LayerNorm(d, d) silently passes d as eps and
    # leaves std around 0.24 instead of 1.0
    report("eps is small (std not crushed)", std, std > 0.9, fmt="{:.4f}")


if __name__ == "__main__":
    test()
