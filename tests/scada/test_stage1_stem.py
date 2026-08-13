"""Stage 1 -- Stem: (value, mask) at each timestep -> d_model features."""

import torch
from models.scada_encoder_tcn import Stem
from helpers import banner, check_shape, report


def test():
    banner("stage 1: stem")
    torch.manual_seed(0)
    N, T, d = 80, 600, 128

    x = torch.randn(N, T)
    mask = (torch.rand(N, T) > 0.05).float()

    stem = Stem(d_model=d)
    h = stem(x, mask)

    # widens 2 features (value, mask) to d_model at every timestep
    check_shape("output shape", h, (N, d, T))

    # kernel_size=1 means no time mixing at all: shifting the input in time
    # must shift the output identically, with nothing bleeding across steps
    xs = torch.roll(x, shifts=5, dims=-1)
    ms = torch.roll(mask, shifts=5, dims=-1)
    stem.eval()
    with torch.no_grad():
        hs = stem(xs, ms)
    drift = (hs[:, :, 5:] - torch.roll(h, 5, -1)[:, :, 5:]).abs().max().item()
    report("no time mixing (kernel_size=1)", drift, drift < 1e-5)

    # deterministic in eval: no dropout or other randomness at this stage
    with torch.no_grad():
        same = torch.equal(stem(x, mask), stem(x, mask))
    report("deterministic", float(same), same, fmt="{:.0f}")


if __name__ == "__main__":
    test()
