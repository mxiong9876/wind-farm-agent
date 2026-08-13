"""Stage 5 -- TCNTrunk: stem plus a dilated stack."""

import torch
from models.scada_encoder_tcn import TCNTrunk
from helpers import banner, check_shape, check_trunk_causal, report


def test():
    banner("stage 5: trunk")
    torch.manual_seed(0)
    N, T, d = 80, 600, 128
    x = torch.randn(N, T)
    m = (torch.rand(N, T) > 0.05).float()

    trunk = TCNTrunk(d_model=d)
    check_shape("output shape", trunk(x, m), (N, d, T))

    # a receptive field shorter than the window means the oldest data is
    # INVISIBLE to the final output -- silently, with no error
    rf = trunk.receptive_field
    report(f"receptive field {rf} covers T={T}", rf, rf >= T, fmt="{:.0f}")

    # nn.ModuleList registers parameters; a plain [] does not, and the model
    # would train with every block frozen while reporting nothing
    n = sum(p.numel() for p in trunk.parameters())
    report("parameters registered (ModuleList)", n, n > 100_000, fmt="{:,.0f}")

    n_blocks = len(list(trunk.blocks))
    report("all 8 blocks present", n_blocks, n_blocks == 8, fmt="{:.0f}")

    check_trunk_causal(trunk)

    # gradients must reach the first block, or the stack is disconnected
    trunk.train()
    trunk(x, m).sum().backward()
    first = next(trunk.blocks[0].parameters())
    g = first.grad.abs().max().item()
    report("gradient reaches block 0", g, g > 0)


if __name__ == "__main__":
    test()
