"""Stage 8 -- subsample to tokens and hand off to transformer axis order."""

import torch
from models.scada_encoder_tcn import ScadaTCNEncoder
from helpers import TOL, banner, check_shape, report


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

    # Token k of channel c sits at flat index c*P + k and summarizes data up to
    # timestep (k+1)*stride - 1. So perturbing t >= cut must leave every token
    # that ends before cut untouched.
    #
    # Through the FULL encoder it does not, and that is not a bug: RevIN
    # normalizes by whole-window statistics, so a change at t=500 moves the mean
    # and std applied at t=100. That number is printed as INFO -- it is what
    # makes this encoder health-path only. The control path needs running
    # normalization, and only then can causality be asserted end to end.
    cut = 500
    safe = cut // stride          # tokens ending at (k+1)*stride - 1 < cut
    x2 = x.clone()
    x2[:, :, cut:] += 10.0

    enc.eval()
    P = enc.out_len
    with torch.no_grad():
        a, b = enc(x, m), enc(x2, m)
    early = torch.zeros(C * P, dtype=torch.bool)
    for c in range(C):
        early[c * P:c * P + safe] = True
    drift = (a[:, early] - b[:, early]).abs().max().item()
    print(f"  {'end-to-end drift (RevIN makes this >0)':<38} {drift:>12.2e}  INFO")

    # The assertion goes through the trunk alone, where causality is actually
    # claimed. This also pins the SUBSAMPLE INDEXING: token `safe` is the first
    # one whose window reaches past the cut, so an off-by-one in the stride
    # offset would either leak here or leave token `safe` suspiciously clean.
    xf, mf = x.reshape(B * C, T), m.reshape(B * C, T)
    with torch.no_grad():
        ha = enc.trunk(xf, mf)[:, :, stride - 1::stride]
        hb = enc.trunk(x2.reshape(B * C, T), mf)[:, :, stride - 1::stride]
    leak = (ha[:, :, :safe] - hb[:, :, :safe]).abs().max().item()
    moved = (ha[:, :, safe] - hb[:, :, safe]).abs().max().item()
    report(f"tokens 0..{safe - 1} unchanged (trunk alone)", leak, leak < TOL)
    report(f"token {safe} does move (boundary is exact)", moved, moved > 1.0,
           fmt="{:.2f}")


if __name__ == "__main__":
    test()
