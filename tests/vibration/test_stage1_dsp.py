"""Vibration stage 1 -- DSP frontend: raw acceleration to a time-frequency image.

This is where a misconfigured frontend produces a perfectly plausible picture
of the wrong thing. Shapes stay right, nothing raises, the conv net trains --
and the band that carried the fault was never in the input.

The headline check here is inner-race vs outer-race discrimination. That is the
whole reason the envelope branch exists, and it is what breaks first if anyone
retunes n_fft or env_fmax without realising the two interact.
"""

import math

import torch

from vibration_encoder_2dconv import VibrationDSP, linear_filterbank
from helpers import banner, check_shape, report

FS = 25600.0
L = 25600


def _peaks(profile, hz, lo, hi, frac=0.5):
    """Local maxima above `frac` of the global peak, within [lo, hi] Hz.

    The threshold matters: without it every ripple in the noise floor counts as
    a peak and the inner/outer race check passes for the wrong reason.
    """
    n = len(profile)
    thr = profile.max() * frac
    return [round(hz[i].item()) for i in range(n)
            if lo < hz[i] < hi and profile[i] > thr
            and profile[i] > profile[max(i - 1, 0)]
            and profile[i] > profile[min(i + 1, n - 1)]]


def test():
    banner("vibration stage 1: DSP frontend")
    torch.manual_seed(0)
    dsp = VibrationDSP()
    n_freq = dsp.n_freq

    fb = dsp.fb_raw
    check_shape("filterbank shape", fb, (n_freq, dsp.n_fft // 2 + 1))
    report("rows sum to 1", (fb.sum(1) - 1).abs().max().item(),
           bool(torch.allclose(fb.sum(1), torch.ones(n_freq), atol=1e-6)))
    report("no empty rows", float((fb.sum(1) > 0).all()),
           bool((fb.sum(1) > 0).all()), fmt="{:.0f}")

    # LINEAR spacing, not mel. Mel is tuned to human hearing and collapses
    # resolution above ~1 kHz; defect frequencies are harmonics of shaft rate
    # and are linear, so mel would smear exactly what we are looking for.
    centres = torch.arange(1, n_freq + 1) * (FS / 2) / (n_freq + 1)
    gaps = centres[1:] - centres[:-1]
    report("filter centres evenly spaced (linear, not mel)",
           (gaps - gaps[0]).abs().max().item(),
           (gaps - gaps[0]).abs().max().item() < 1e-3)

    # a filter narrower than the STFT bin spacing catches no bins at all and
    # that band goes invisible, silently. Construction must refuse.
    try:
        linear_filterbank(64, 1024, FS, 0.0, 500.0)   # 20 bins, 64 filters
        raised = False
    except AssertionError:
        raised = True
    report("starved filterbank raises at construction", float(raised), raised,
           fmt="{:.0f}")

    z = torch.stft(torch.randn(2, L), n_fft=dsp.n_fft, hop_length=dsp.hop,
                   window=torch.hann_window(dsp.n_fft), center=False,
                   return_complex=True)
    report("n_frames formula matches torch.stft", dsp.n_frames,
           dsp.n_frames == z.shape[-1], fmt="{:.0f}")

    # --- envelope demodulation -------------------------------------------
    # A bearing spall does not ring at its own defect frequency. It hammers a
    # high-frequency structural resonance once per pass and amplitude-modulates
    # it at the defect rate. So: a 5 kHz carrier modulated at 137 Hz must show
    # up at 5 kHz in the raw plane and at 137 Hz in the envelope plane.
    t = torch.arange(L) / FS
    car, defect = 5000.0, 137.0
    am = (1 + 0.8 * torch.cos(2 * math.pi * defect * t)) * torch.cos(2 * math.pi * car * t)
    img, level = dsp(am[None, :])
    check_shape("dsp image (raw + envelope planes)", img,
                (1, 2, n_freq, dsp.n_frames))

    raw_hz = torch.arange(1, n_freq + 1) * (FS / 2) / (n_freq + 1)
    env_hz = torch.arange(1, n_freq + 1) * 600.0 / (n_freq + 1)
    raw_pk = raw_hz[img[0, 0].mean(1).argmax()].item()
    env_pk = env_hz[img[0, 1].mean(1).argmax()].item()
    report("raw plane peaks at the carrier", abs(raw_pk - car),
           abs(raw_pk - car) < (FS / 2) / (n_freq + 1), fmt="{:.0f}")
    report("envelope plane peaks at the defect rate", abs(env_pk - defect),
           abs(env_pk - defect) < 600.0 / (n_freq + 1), fmt="{:.1f}")

    # THE diagnosis. An inner-race fault rotates through the load zone once per
    # revolution, so its envelope line carries shaft-rate sidebands and appears
    # as a TRIPLET. An outer-race fault is stationary in the load zone and gives
    # a single line. At 1500 rpm the triplet is spaced 25 Hz, so any settings
    # coarser than that merge all three and report inner-race damage as
    # outer-race. This is what n_fft=5120 / env_fmax=600 buys.
    inner = (1 + 0.6 * torch.cos(2 * math.pi * 240 * t)
             + 0.3 * torch.cos(2 * math.pi * 215 * t)
             + 0.3 * torch.cos(2 * math.pi * 265 * t)) * torch.cos(2 * math.pi * car * t)
    outer = (1 + 0.6 * torch.cos(2 * math.pi * 160 * t)) * torch.cos(2 * math.pi * car * t)
    p_in = _peaks(dsp(inner[None, :])[0][0, 1].mean(1), env_hz, 180, 300)
    p_out = _peaks(dsp(outer[None, :])[0][0, 1].mean(1), env_hz, 100, 300)
    report("inner race resolves as a triplet", len(p_in), len(p_in) == 3,
           fmt="{:.0f}")
    report("outer race resolves as a single line", len(p_out), len(p_out) == 1,
           fmt="{:.0f}")
    report("the two faults stay distinguishable", abs(len(p_in) - len(p_out)),
           len(p_in) != len(p_out), fmt="{:.0f}")

    # cross-check the analytic signal against a reference implementation
    try:
        import numpy as np
        from scipy.signal import hilbert
        xn = am[None, :] - am.mean()
        xn = xn / xn.pow(2).mean().sqrt()
        sig = xn[0].numpy().astype(np.float64)
        fr = np.fft.fftfreq(L, d=1 / FS)
        keep = (np.abs(fr) >= 2000) & (np.abs(fr) <= 10000)   # SYMMETRIC -> real
        ref = np.abs(hilbert(np.real(np.fft.ifft(np.fft.fft(sig) * keep))))
        ours = torch.fft.ifft(torch.fft.fft(xn) * dsp.analytic).abs()[0].numpy()
        err = float(np.abs(ours - ref).max() / ref.max())
        report("analytic envelope matches scipy.hilbert", err, err < 1e-3)
    except ImportError:
        print(f"  {'analytic envelope vs scipy':<38} {'skipped':>12}  INFO")

    # energy outside the resonance band must not reach the envelope at all
    oob = torch.cos(2 * math.pi * 500.0 * t)[None, :]
    leak = torch.fft.ifft(torch.fft.fft(oob) * dsp.analytic).abs().max().item()
    report("band-pass rejects out-of-band energy", leak, leak < 1e-2)

    # --- normalization ----------------------------------------------------
    x = torch.randn(6, L) * 0.5
    i0, l0 = dsp(x)
    i1, l1 = dsp(x * 7.0)
    i2, l2 = dsp(x + 3.0)
    # portability across sensor sensitivities and turbine sizes
    report("spectrogram invariant to 7x gain", (i0 - i1).abs().max().item(),
           (i0 - i1).abs().max().item() < 1e-4)
    report("spectrogram invariant to DC offset", (i0 - i2).abs().max().item(),
           (i0 - i2).abs().max().item() < 1e-4)
    # ...but broadband level is itself a health indicator (ISO 10816), so it
    # must survive somewhere rather than being normalized into oblivion
    report("level token tracks gain", (l1 - l0).abs().min().item(),
           (l1 - l0).abs().min().item() > 0.5, fmt="{:.3f}")
    report("level token ignores DC offset", (l2 - l0).abs().max().item(),
           (l2 - l0).abs().max().item() < 1e-4)

    n_par = sum(1 for _ in dsp.parameters())
    report("dsp has no learnable parameters", n_par, n_par == 0, fmt="{:.0f}")


if __name__ == "__main__":
    test()
