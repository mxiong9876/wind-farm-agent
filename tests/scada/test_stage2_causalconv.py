"""Stage 2 -- CausalConv: mixes across time, but only backwards."""

import torch
from models.scada_encoder_tcn import CausalConv
from helpers import banner, check_shape, check_causal


def test():
    banner("stage 2: causal conv")
    torch.manual_seed(0)
    N, T, d = 80, 600, 128
    h = torch.randn(N, d, T)

    conv = CausalConv(d_model=d)
    check_shape("length preserved", conv(h), (N, d, T))
    check_causal(conv, label="causal (dilation=1)")

    # left-padding must scale with dilation, or length breaks at high dilation
    for dil in (1, 2, 4, 8, 16, 32, 64, 128):
        c = CausalConv(d_model=d, dilation=dil)
        check_shape(f"length @ dilation={dil}", c(h), (N, d, T))
        check_causal(c, label=f"causal @ dilation={dil}")


if __name__ == "__main__":
    test()
