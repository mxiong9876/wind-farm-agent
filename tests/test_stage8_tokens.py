"""Stage 8 -- subsample to tokens and hand off to transformer axis order."""

import torch
from scada_encoder_tcn import ScadaTCNEncoder
from helpers import banner, check_shape, report


def test():
    banner("stage 8: tokens")
    torch.manual_seed(0)
    B, C, T, d, stride = 4, 20, 600, 128, 15

    enc = ScadaTCNEncoder(d_model=d, n_channels=C, context_len=T, stride=stride)
    x = torch.randn(B, C, T)
    m = (torch.rand(B, C, T) > 0.05).float()
    out = enc(x, m)

    expected_len = len(range(stride - 1, T, stride))
    report("out_len matches stride", enc.out_len,
           enc.out_len == expected_len, fmt="{:.0f}")

    # features LAST, then flattened: conv layout (B, C, d, P) -> transformer
    # layout (B, C, P, d) -> flat sequence (B, C*P, d)
    check_shape("flattened token sequence", out, (B, C * enc.out_len, d))

    # tokens PER CHANNEL depend on stride only; the flat length scales with C
    e2 = ScadaTCNEncoder(d_model=d, n_channels=13, context_len=T, stride=stride)
    o2 = e2(torch.randn(B, 13, T), torch.ones(B, 13, T))
    report("tokens per channel independent of C", o2.shape[1] // 13,
           o2.shape[1] // 13 == enc.out_len, fmt="{:.0f}")

    # token k of channel c sits at flat index c*P + k, and ends at timestep
    # k*stride + stride - 1. Perturbing t >= 500 leaves every token ending
    # before 500 untouched IN THE TRUNK -- but RevIN normalizes by whole-window
    # statistics, so the printed number is >0 by design, not by bug. This is
    # what makes the encoder health-path only; the control path needs running
    # normalization.
    P = enc.out_len
    early = torch.zeros(C * P, dtype=torch.bool)
    for c in range(C):
        early[c * P:c * P + safe] = True
    leak = (a[:, early] - b[:, early]).abs().max().item()
    print(f"  {'tokens before cut (RevIN makes this >0)':<38} {leak:>12.2e}  INFO")


if __name__ == "__main__":
    test()
