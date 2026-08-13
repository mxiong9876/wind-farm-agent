"""Vibration stage 3 -- token contract, and interop with the SCADA path.

The output here has to match ScadaTCNEncoder's layout exactly: a channel-major
(B, S, 128) sequence with the channel embedding already added. If the two
modalities disagree on width or ordering, fusion still runs and still trains --
on scrambled input.

This is the only vibration stage that touches scada_encoder_tcn, and only to
borrow the shared PerceiverResampler.
"""

import torch

from models.vibration_encoder_2dconv import VibrationConv2dEncoder
from helpers import banner, check_shape, report

L = 25600


def test():
    banner("vibration stage 3: token contract")
    torch.manual_seed(0)
    B = 4

    enc = VibrationConv2dEncoder().eval()
    C, P, d = enc.n_channels, enc.out_len, enc.d_model
    x = torch.randn(B, C, L) * 0.5
    m = torch.ones(B, C)
    m[0, 3] = 0.0                                    # one dead accelerometer

    # d_model must match SCADA or the shared resampler cannot take both
    report("d_model matches SCADA width", d, d == 128, fmt="{:.0f}")
    with torch.no_grad():
        out = enc(x, m)
    check_shape("C*P spectrogram + C level tokens", out, (B, C * P + C, d))
    report("dead channel keeps output finite", float(torch.isfinite(out).all()),
           bool(torch.isfinite(out).all()), fmt="{:.0f}")

    # an all-missing snapshot must not divide by zero anywhere in the frontend
    with torch.no_grad():
        allz = enc(torch.zeros(1, C, L), torch.zeros(1, C))
    report("all-zero, all-dead stays finite", float(torch.isfinite(allz).all()),
           bool(torch.isfinite(allz).all()), fmt="{:.0f}")

    # a dead sensor and a silent-but-working one both give an all-zero
    # spectrogram, so the mask has to reach the level token or they collapse
    xq = x.clone()
    xq[0, 3] = 0.0
    with torch.no_grad():
        lv_dead = enc(x, m)[0, C * P + 3]
        lv_quiet = enc(xq, torch.ones(B, C))[0, C * P + 3]
    gap = (lv_dead - lv_quiet).abs().max().item()
    report("dead vs silent-but-alive distinguishable", gap, gap > 1e-3,
           fmt="{:.3f}")

    # channel-major layout, matching ScadaTCNEncoder: channel c owns the
    # contiguous slice [c*P, (c+1)*P). Feed two channels identical data --
    # their tokens must still differ, by a CONSTANT offset (the embedding).
    xd = torch.randn(B, C, L)
    xd[:, 1] = xd[:, 0]
    with torch.no_grad():
        od = enc(xd, torch.ones(B, C))
    delta = od[:, 0:P] - od[:, P:2 * P]
    report("identical data, channels still differ", delta.abs().max().item(),
           delta.abs().max().item() > 1e-3, fmt="{:.3f}")
    spread = (delta - delta.mean(dim=1, keepdim=True)).abs().max().item()
    report("channel embedding is a constant offset", spread, spread < 1e-3)

    # the conv trunk runs per accelerometer and must never mix them
    xa = torch.randn(B, C, L)
    xb = xa.clone()
    xb[:, 3] += 5.0
    with torch.no_grad():
        oa, ob = enc(xa, torch.ones(B, C)), enc(xb, torch.ones(B, C))
    keep = torch.ones(C * P + C, dtype=torch.bool)
    keep[3 * P:4 * P] = False
    keep[C * P + 3] = False
    bleed = (oa[:, keep] - ob[:, keep]).abs().max().item()
    report("accelerometers stay independent", bleed, bleed < 1e-4)

    # tokens must be reasonably scaled or the resampler's LayerNorms start
    # from a bad place
    report("token scale well conditioned", out.std().item(),
           out.std().item() < 5 and out.abs().max().item() < 50, fmt="{:.3f}")

    # gradients must reach every parameter
    enc2 = VibrationConv2dEncoder()
    enc2.train()
    head = torch.nn.Linear(128, 1)
    y = torch.randint(0, 2, (B,)).float()
    torch.nn.functional.binary_cross_entropy_with_logits(
        head(enc2(x, m).mean(1)).squeeze(-1), y).backward()
    nog = [n for n, p in enc2.named_parameters() if p.grad is None]
    dead = [n for n, p in enc2.named_parameters()
            if p.grad is not None and p.grad.abs().max().item() == 0.0]
    if nog:
        print(f"  no grad at all: {nog}")
    if dead:
        print(f"  zero grad: {dead}")
    report("every parameter has a gradient", len(nog), not nog, fmt="{:.0f}")
    report("no parameter is dead", len(dead), not dead, fmt="{:.0f}")

    # fixed transforms must not bloat the checkpoint or pin it to one config
    dsp_keys = [k for k in enc.state_dict() if k.startswith("dsp.")]
    report("dsp buffers stay out of the checkpoint", len(dsp_keys),
           not dsp_keys, fmt="{:.0f}")

    # --- interop with the SCADA resampler ---------------------------------
    from models.scada_encoder_tcn import PerceiverResampler
    r = PerceiverResampler(d_model=128, n_latents=32).eval()
    with torch.no_grad():
        check_shape("vibration -> shared resampler", r(out), (B, 32, 128))
    # a turbine with fewer accelerometers must still reach fusion as 32 latents
    for c in (4, 2):
        e = VibrationConv2dEncoder(n_channels=c).eval()
        with torch.no_grad():
            check_shape(f"C={c} -> fixed latents", r(e(torch.randn(2, c, L))),
                        (2, 32, 128))

    # --- guards -----------------------------------------------------------
    cases = [
        ("wrong channel count", lambda: enc(torch.randn(2, 5, L))),
        ("wrong snapshot length", lambda: enc(torch.randn(2, C, 12000))),
        ("last stage != d_model",
         lambda: VibrationConv2dEncoder(channels=(64, 64, 64, 64))),
        ("stage/stride mismatch",
         lambda: VibrationConv2dEncoder(strides=((2, 2),))),
        ("resonance band above Nyquist",
         lambda: VibrationConv2dEncoder(env_band=(2000.0, 30000.0))),
    ]
    for label, fn in cases:
        try:
            fn()
            ok = False
        except (AssertionError, RuntimeError):
            ok = True
        report(f"rejects: {label}", float(ok), ok, fmt="{:.0f}")


if __name__ == "__main__":
    test()
