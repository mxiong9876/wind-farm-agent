"""Stage 10 -- flatten, side tokens, and the Perceiver resampler.

Part A: (B, C, P, d) grid -> (B, S, d) list
Part B: append categorical and static tokens
Part C: compress a variable-length sequence to a fixed 32 latents
"""

import torch
from models.common import PerceiverResampler, ResamplerLayer
from helpers import banner, check_shape, report


def test_resampler():
    banner("stage 10c: perceiver resampler")
    torch.manual_seed(0)
    B, d, L = 4, 128, 32

    r = PerceiverResampler(d_model=d, n_latents=L).eval()

    # THE point of the component: output size is fixed no matter the input
    # length, so a turbine with fewer sensors still yields L tokens and the
    # fusion module never has to know how many channels fed it
    for S in (804, 500, 64, 2000):
        check_shape(f"S={S} -> fixed latents", r(torch.randn(B, S, d)), (B, L, d))

    # latents must be trainable, or they stay frozen random noise forever
    trainable = r.latents.requires_grad
    report("latents are nn.Parameter", float(trainable), trainable, fmt="{:.0f}")

    # torch.rand leaves every latent ~75% aligned with every other, so they
    # receive near-identical gradients and cannot specialize. randn gives
    # near-orthogonal starts.
    n = torch.nn.functional.normalize(r.latents, dim=-1)
    sim = n @ n.T
    off = sim[~torch.eye(L, dtype=torch.bool)]
    report("latents near-orthogonal at init (randn not rand)",
           off.mean().item(), abs(off.mean().item()) < 0.2, fmt="{:+.3f}")

    # content must actually reach the output -- if attention were disconnected
    # the latents would return the same thing for any input
    with torch.no_grad():
        a = r(torch.randn(B, 804, d))
        b = r(torch.randn(B, 804, d))
    diff = (a - b).abs().max().item()
    report("output depends on input", diff, diff > 1e-3, fmt="{:.4f}")

    # attention is permutation-invariant over keys: reordering tokens must NOT
    # change the result. Order carries no meaning here -- position information
    # lives inside the token values (channel embedding, time encoding).
    x = torch.randn(B, 804, d)
    perm = torch.randperm(804)
    with torch.no_grad():
        same = (r(x) - r(x[:, perm])).abs().max().item()
    report("permutation invariant over tokens", same, same < 1e-3)

    # NOTE: at initialization the resampler behaves close to an AVERAGING
    # operation -- attention weights are near-uniform, so one token out of 804
    # carries ~1/804 of the mass, and norm_x flattens outliers besides.
    # Perturbing a single token moves the output by ~1e-6. Sharp attention to
    # individual anomalous readings is something training must LEARN; it is not
    # available for free. Worth revisiting (more latents, more layers) if the
    # health head struggles to flag single-sensor anomalies.
    x2 = x.clone()
    x2[:, 17] += 50.0
    with torch.no_grad():
        one = (r(x) - r(x2)).abs().max().item()
    print(f"  {'single-token sensitivity at init':<38} {one:>12.2e}  INFO")

    # replacing a quarter of the tokens outright must move the output. Note
    # this REPLACES content rather than shifting it: norm_x normalizes every
    # token before attention, so adding a constant or scaling all tokens is
    # invisible by design (verified below).
    x3 = x.clone()
    x3[:, :200] = torch.randn(B, 200, d) * 2.0
    with torch.no_grad():
        many = (r(x) - r(x3)).abs().max().item()
    report("responds to replaced token content", many, many > 1e-2, fmt="{:.4f}")

    # the flip side of that normalization: a uniform offset or rescale of the
    # whole sequence carries no information and is correctly ignored
    with torch.no_grad():
        shifted = (r(x) - r(x + 5.0)).abs().max().item()
        scaled = (r(x) - r(x * 3.0)).abs().max().item()
    report("invariant to uniform offset (norm_x)", shifted, shifted < 1e-4)
    report("invariant to uniform rescale (norm_x)", scaled, scaled < 1e-4)

    # gradients must reach the latents
    r.train()
    r(torch.randn(B, 804, d)).sum().backward()
    g = r.latents.grad.abs().max().item()
    report("gradient reaches latents", g, g > 0)

    # batch_first=True: with the legacy default the batch axis is read as the
    # sequence, which runs fine and produces nonsense. Asymmetric shape catches it.
    r2 = PerceiverResampler(d_model=d, n_latents=L).eval()
    with torch.no_grad():
        out = r2(torch.randn(3, 777, d))
    check_shape("batch_first (asymmetric B vs S)", out, (3, L, d))

    # n_heads must divide d_model
    try:
        ResamplerLayer(d_model=128, n_heads=7)
        ok = False
    except Exception:
        ok = True
    report("rejects n_heads not dividing d_model", float(ok), ok, fmt="{:.0f}")


def test_encoder_tail(enc_cls=None):
    """Parts A and B, through the full encoder.

    Skipped automatically if ScadaTCNEncoder does not yet accept the side
    inputs, so this file stays runnable mid-build.
    """
    banner("stage 10a/b: flatten and side tokens")
    torch.manual_seed(0)
    B, C, T, d = 4, 20, 600, 128

    from models.scada_encoder_tcn import ScadaTCNEncoder
    try:
        enc = ScadaTCNEncoder(d_model=d, n_channels=C, context_len=T,
                              n_static=7).eval()
    except TypeError as e:
        print(f"  skipped: encoder does not take n_static yet ({e})")
        return

    x = torch.randn(B, C, T)
    m = torch.ones(B, C, T)
    cat = torch.stack([torch.randint(0, c, (B,)) for c in (12, 256, 2)], dim=1)
    st = torch.randn(B, 7)

    P = enc.out_len
    base = C * P

    # each optional input must extend the sequence independently, so all four
    # combinations have to work -- a block that reads `h` instead of `seq`
    # passes three of these and fails the fourth
    with torch.no_grad():
        check_shape("no side inputs", enc(x, m), (B, base, d))
        check_shape("categorical only", enc(x, m, cat), (B, base + 3, d))
        check_shape("static only", enc(x, m, None, st), (B, base + 1, d))
        check_shape("both", enc(x, m, cat, st), (B, base + 4, d))

    # flatten must preserve channel-major order: channel c's tokens occupy
    # the contiguous slice [c*P, (c+1)*P)
    xd = torch.randn(B, C, T)
    xd[:, 1] = xd[:, 0]
    with torch.no_grad():
        flat = enc(xd, torch.ones(B, C, T))
    ch0 = flat[:, 0:P]
    ch1 = flat[:, P:2 * P]
    diff = (ch0 - ch1).abs().max().item()
    report("channel-major layout preserved", diff, diff > 1e-3, fmt="{:.4f}")

    # categorical tokens must actually depend on the codes passed in
    cat2 = cat.clone()
    cat2[:, 1] = (cat2[:, 1] + 7) % 256
    with torch.no_grad():
        a, b = enc(x, m, cat, st), enc(x, m, cat2, st)
    moved = (a[:, base:base + 3] - b[:, base:base + 3]).abs().max().item()
    report("alarm code changes its token", moved, moved > 1e-3, fmt="{:.4f}")

    # and the static token must depend on the static features
    with torch.no_grad():
        a, b = enc(x, m, cat, st), enc(x, m, cat, st + 1.0)
    moved = (a[:, -1] - b[:, -1]).abs().max().item()
    report("static features change their token", moved, moved > 1e-3, fmt="{:.4f}")


def test():
    test_resampler()
    test_encoder_tail()


if __name__ == "__main__":
    test()
