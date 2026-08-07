"""Stage 7 -- channel folding: (B, C, T) <-> (B*C, T).

The failure mode here is silent. If fold and unfold disagree, shapes still
look right, causality still passes, loss still decreases -- the model just
learns from scrambled data.
"""

import torch
from scada_encoder_tcn import ScadaTCNEncoder
from helpers import banner, check_shape, report


def test():
    banner("stage 7: channel fold")
    torch.manual_seed(0)
    B, C, T, d = 4, 20, 600, 128

    # arange gives every element a unique value, so a scramble is detectable
    x = torch.arange(B * C * T, dtype=torch.float32).reshape(B, C, T)
    xf = x.reshape(B * C, T)

    ok = torch.equal(xf.reshape(B, C, T), x)
    report("round trip preserves data", float(ok), ok, fmt="{:.0f}")

    # row b*C + c must be exactly x[b, c]
    hits = all(torch.equal(xf[b * C + c], x[b, c])
               for b in range(B) for c in range(C))
    report("row b*C+c maps to [b, c]", float(hits), hits, fmt="{:.0f}")

    # end to end through the encoder
    enc = ScadaTCNEncoder(d_model=d, n_channels=C, context_len=T)
    m = (torch.rand(B, C, T) > 0.05).float()
    out = enc(torch.randn(B, C, T), m)
    check_shape("encoder output", out, (B, C * enc.out_len, d))

    # channels must stay independent: perturbing channel 3 must not move any
    # other channel, since the trunk never mixes across sensors. After the
    # flatten, channel c occupies the contiguous slice [c*P, (c+1)*P).
    enc.eval()
    P = enc.out_len
    xa = torch.randn(B, C, T)
    xb = xa.clone()
    xb[:, 3] += 10.0
    with torch.no_grad():
        oa, ob = enc(xa, torch.ones(B, C, T)), enc(xb, torch.ones(B, C, T))
    keep = torch.ones(C * P, dtype=torch.bool)
    keep[3 * P:4 * P] = False
    bleed = (oa[:, keep] - ob[:, keep]).abs().max().item()
    report("channels stay independent", bleed, bleed < 1e-4)


if __name__ == "__main__":
    test()
