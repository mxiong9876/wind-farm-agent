"""Vibration stage 2 -- 2D conv trunk over the (frequency, time) plane.

Two failure modes live here, and neither announces itself:

  1. a stride/shortcut mismatch, which only shows up at grid sizes where the
     two output formulas happen to disagree
  2. the wrong normalization, which trains fine and quietly makes one turbine's
     prediction depend on whichever turbines shared its batch
"""

import torch

from vibration_encoder_2dconv import ConvBlock2d, VibrationConv2dEncoder
from helpers import banner, report

L = 25600


def test():
    banner("vibration stage 2: 2D conv trunk")
    torch.manual_seed(0)

    # 3x3/pad-1/stride-s and 1x1/pad-0/stride-s both give floor((n-1)/s)+1. If
    # they ever diverge the residual add throws, so check odd sizes and 1x1
    # grids where off-by-one errors surface.
    bad = []
    for c_in, c_out, s in [(32, 64, (2, 1)), (64, 128, (2, 2)),
                           (128, 128, (2, 2)), (16, 16, (1, 1)), (32, 32, (3, 3))]:
        blk = ConvBlock2d(c_in, c_out, s, n_groups=8).eval()
        for f, t in [(64, 17), (13, 5), (7, 7), (1, 1), (5, 3)]:
            want = (2, c_out, (f - 1) // s[0] + 1, (t - 1) // s[1] + 1)
            got = tuple(blk(torch.randn(2, c_in, f, t)).shape)
            if got != want:
                bad.append((c_in, c_out, s, f, t, got, want))
    if bad:
        print(f"  mismatches: {bad[:3]}")
    report("trunk and shortcut agree at every size", len(bad), not bad,
           fmt="{:.0f}")

    # GroupNorm, not BatchNorm. With BatchNorm a turbine's prediction would
    # depend on whichever other turbines shared its batch, and eval would switch
    # to running estimates gathered from whatever mix of healthy and faulty
    # machines happened to train it. Feeding a sample alone and inside a batch
    # must give identical results.
    enc = VibrationConv2dEncoder().eval()
    C = enc.n_channels
    xb = torch.randn(4, C, L)
    with torch.no_grad():
        alone = enc(xb[:1], torch.ones(1, C))
        in_batch = enc(xb, torch.ones(4, C))[:1]
    drift = (alone - in_batch).abs().max().item()
    report("batch-independent normalization", drift, drift < 1e-4)

    enc.train()
    with torch.no_grad():
        differs = not torch.equal(enc(xb, torch.ones(4, C)),
                                  enc(xb, torch.ones(4, C)))
    report("dropout active in train()", float(differs), differs, fmt="{:.0f}")
    enc.eval()
    with torch.no_grad():
        same = torch.equal(enc(xb, torch.ones(4, C)), enc(xb, torch.ones(4, C)))
    report("dropout inert in eval()", float(same), same, fmt="{:.0f}")


if __name__ == "__main__":
    test()
