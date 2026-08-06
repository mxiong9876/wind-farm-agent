"""Stage 4 -- CausalBlock: two convs, norms, activations, plus a residual."""

import torch
from scada_encoder_tcn import CausalBlock
from helpers import banner, check_shape, check_causal, report


def test():
    banner("stage 4: causal block")
    torch.manual_seed(0)
    N, T, d = 80, 600, 128
    h = torch.randn(N, d, T)

    block = CausalBlock(d_model=d, dilation=1)
    check_shape("shape preserved (residual needs it)", block(h), (N, d, T))

    for dil in (1, 2, 4, 8, 16, 32, 64, 128):
        check_causal(CausalBlock(d_model=d, dilation=dil),
                     label=f"causal @ dilation={dil}")

    # the residual add must actually be there: zeroing both conv weights should
    # leave the block close to the identity, not close to zero
    b = CausalBlock(d_model=d, dilation=1).eval()
    with torch.no_grad():
        for p in b.parameters():
            p.zero_()
        out = b(h)
    err = (out - h).abs().max().item()
    report("residual present (identity when zeroed)", err, err < 1e-5)

    # dropout must be active in train() and inert in eval()
    b2 = CausalBlock(d_model=d, dilation=1)
    b2.train()
    with torch.no_grad():
        differs = not torch.equal(b2(h), b2(h))
    report("dropout active in train()", float(differs), differs, fmt="{:.0f}")
    b2.eval()
    with torch.no_grad():
        same = torch.equal(b2(h), b2(h))
    report("dropout inert in eval()", float(same), same, fmt="{:.0f}")


if __name__ == "__main__":
    test()
