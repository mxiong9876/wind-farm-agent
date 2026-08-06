"""
in : x    (B, C, L) float32   raw acceleration, one row per accelerometer
     mask (B, C)    float32   1 = channel healthy, 0 = dead / invalid snapshot
out:      (B, C*P + C, 128) float32   tokens

B = Batch    - how many snapshots you process at once
C = Channels - how many accelerometers
L = Length   - raw samples in each snapshot (1 s at 25.6 kHz = 25600)
P = time tokens produced per channel by the conv stack

Vibration is a burst modality: the CMS wakes up, grabs a short high-rate
snapshot, and sleeps. So unlike SCADA there is no long context to march
through causally -- the whole snapshot is one observation, and the useful
structure is in the time-FREQUENCY plane. Hence a DSP frontend followed by a
2D conv net over (frequency, time), rather than a TCN over raw samples.

The output contract deliberately matches scada_encoder_tcn.ScadaTCNEncoder:
a variable-length (B, S, d_model) sequence at d_model=128, channel-major, with
a learned channel embedding already added. Both encoders therefore feed the
SAME PerceiverResampler and arrive at fusion as (B, 32, 128).

    from scada_encoder_tcn import PerceiverResampler
    tokens = VibrationConv2dEncoder()(x, mask)          # (B, 48, 128)
    latents = PerceiverResampler(d_model=128)(tokens)   # (B, 32, 128)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DSP frontend -- fixed transforms, zero learnable parameters
# ---------------------------------------------------------------------------
def linear_filterbank(n_freq, n_fft, fs, f_min=0.0, f_max=None):
    """Triangular filterbank on a LINEAR frequency axis.

    Deliberately not mel. Mel spacing is tuned to human hearing and collapses
    resolution above ~1 kHz. Machinery faults sit at exact harmonics of shaft
    rate and at bearing defect frequencies, all linear in frequency, so mel
    would smear away precisely the structure being looked for.
    """
    f_max = fs / 2 if f_max is None else f_max
    assert 0.0 <= f_min < f_max <= fs / 2, \
        f"band [{f_min}, {f_max}] Hz invalid for fs={fs}"

    n_bins = n_fft // 2 + 1
    bin_f = torch.linspace(0.0, fs / 2, n_bins)          # (n_bins,)
    edges = torch.linspace(f_min, f_max, n_freq + 2)     # (n_freq + 2,)

    lo, ctr, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rise = (bin_f[None, :] - lo) / (ctr - lo)
    fall = (hi - bin_f[None, :]) / (hi - ctr)
    fb = torch.clamp(torch.minimum(rise, fall), min=0.0)  # (n_freq, n_bins)

    # An all-zero row means that filter fell between two STFT bins: the band is
    # invisible to the model, and nothing downstream would ever complain. Fail
    # at construction instead.
    rows = fb.sum(dim=1)
    assert bool((rows > 0).all()), (
        f"{int((rows <= 0).sum())} of {n_freq} filters are empty over "
        f"[{f_min}, {f_max}] Hz at n_fft={n_fft} (bin spacing {fs / n_fft:.1f} Hz). "
        f"Raise n_fft, widen the band, or lower n_freq."
    )
    return fb / rows[:, None]


class VibrationDSP(nn.Module):
    """Raw acceleration -> a 2-channel time-frequency image, plus a level scalar.

    plane 0: log band-spectrogram of the raw signal
    plane 1: log band-spectrogram of the demodulated ENVELOPE

    The envelope plane is what finds rolling-element bearing faults. A spall
    does not ring at its own defect frequency; it hammers a high-frequency
    structural resonance once per pass and amplitude-modulates that resonance
    at the defect rate. In a plain spectrum this is a smear of energy up at
    2-10 kHz with no obvious periodicity. After band-passing to the resonance
    and taking the analytic envelope, the defect rate and its harmonics stand
    out as ordinary spectral lines. Both planes share one (n_freq, n_frames)
    grid so they stack as input channels to the conv net.
    """

    def __init__(self, fs=25600.0, snapshot_len=25600, n_fft=5120, hop=1280,
                 n_freq=64, env_band=(2000.0, 10000.0), env_fmax=600.0,
                 eps=1e-8):
        super().__init__()
        assert snapshot_len >= n_fft, \
            f"snapshot {snapshot_len} shorter than one STFT window {n_fft}"
        assert 0.0 < env_band[0] < env_band[1] < fs / 2, \
            f"resonance band {env_band} must sit inside (0, {fs / 2}) Hz"

        self.fs = fs
        self.snapshot_len = snapshot_len
        self.n_fft = n_fft
        self.hop = hop
        self.n_freq = n_freq
        self.eps = eps
        # center=False, so no edge padding and the count is exact
        self.n_frames = 1 + (snapshot_len - n_fft) // hop

        # RESOLUTION. Two stages blur the envelope spectrum, in series, and the
        # coarser one wins -- so they have to be set together or neither helps:
        #
        #   STFT bin width    = fs / n_fft        = 5.0 Hz here
        #   band spacing      = env_fmax/(n_freq+1) = 9.2 Hz here
        #
        # What this has to resolve is shaft-rate sidebands. An inner-race fault
        # rotates through the load zone once per revolution, so its envelope
        # line is modulated at shaft rate and appears as a TRIPLET
        # (BPFI - fr, BPFI, BPFI + fr). An outer-race fault is stationary in the
        # load zone and gives a single line. At 1500 rpm that triplet is spaced
        # 25 Hz, so bands wider than ~25 Hz merge all three and report an
        # inner-race fault as an outer-race one.
        #
        # The earlier defaults (n_fft=1024, env_fmax=2000) gave 25 Hz bins and
        # 30.8 Hz bands: the triplet collapsed to one peak. Raising n_fft alone
        # did nothing, because the filterbank threw the extra detail back out.
        # The fix is both: a 200 ms window for 5 Hz bins, and 64 bands spent on
        # [0, 600] Hz -- where defect orders actually live -- instead of being
        # spread across 2 kHz of mostly empty spectrum.
        #
        # Cost: a longer window means fewer frames (17, not 97). That is the
        # right trade here. A bearing fault signature is stationary across a
        # one-second snapshot, so time resolution buys nothing while frequency
        # resolution buys the diagnosis.

        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("fb_raw",
                             linear_filterbank(n_freq, n_fft, fs),
                             persistent=False)
        # the envelope of a band-passed signal is low-frequency by construction:
        # defect rates and their first harmonics live below ~600 Hz, so the
        # envelope plane spends every band there rather than on empty spectrum
        self.register_buffer("fb_env",
                             linear_filterbank(n_freq, n_fft, fs, 0.0, env_fmax),
                             persistent=False)

        # One mask does band-pass AND Hilbert at once: zero the negative half,
        # double the positive half, keep only the resonance band. Multiplying
        # the spectrum by this and inverting gives the analytic signal of the
        # band-passed waveform, whose modulus is the envelope.
        freqs = torch.fft.fftfreq(snapshot_len, d=1.0 / fs)
        band = (freqs >= env_band[0]) & (freqs <= env_band[1])
        self.register_buffer("analytic", 2.0 * band.to(torch.float32),
                             persistent=False)

    def _spec(self, sig, fb):
        # sig: (N, L) real -> (N, n_freq, n_frames)
        z = torch.stft(sig, n_fft=self.n_fft, hop_length=self.hop,
                       win_length=self.n_fft, window=self.window,
                       center=False, return_complex=True)
        return torch.log1p(fb @ z.abs())

    def forward(self, sig, mask=None):
        # sig: (N, L) where N = B*C. mask: (N,) or None.
        assert sig.shape[-1] == self.snapshot_len, \
            f"got {sig.shape[-1]} samples, built for {self.snapshot_len}"
        if mask is not None:
            sig = sig * mask[:, None]        # a dead channel goes exactly silent

        sig = sig - sig.mean(dim=-1, keepdim=True)   # kill accelerometer DC offset
        rms = sig.pow(2).mean(dim=-1, keepdim=True).sqrt()          # (N, 1)
        xn = sig / rms.clamp(min=self.eps)

        env = torch.fft.ifft(torch.fft.fft(xn, dim=-1) * self.analytic,
                             dim=-1).abs()
        env = env - env.mean(dim=-1, keepdim=True)   # carrier DC holds no defect info

        img = torch.stack([self._spec(xn, self.fb_raw),
                           self._spec(env, self.fb_env)], dim=1)
        return img, torch.log1p(rms)     # (N, 2, n_freq, n_frames), (N, 1)


# ---------------------------------------------------------------------------
# 2D conv trunk
# ---------------------------------------------------------------------------
class ConvBlock2d(nn.Module):
    """Two 3x3 convs over (freq, time), norms, activations, plus a residual."""

    def __init__(self, c_in, c_out, stride=(2, 2), n_groups=8, dropout=0.1):
        super().__init__()
        assert c_out % n_groups == 0, \
            f"{c_out} channels not divisible by {n_groups} groups"
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1)
        # GroupNorm, not BatchNorm: statistics must not mix turbines inside a
        # batch, and eval must not depend on running estimates gathered from
        # whatever mixture of healthy and faulty machines happened to train it
        self.norm1 = nn.GroupNorm(n_groups, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(n_groups, c_out)
        self.drop = nn.Dropout(dropout)
        # 3x3/pad-1/stride-s and 1x1/pad-0/stride-s both output floor((n-1)/s)+1,
        # so the shortcut lines up with the trunk at every input size
        self.proj = (nn.Identity() if c_in == c_out and tuple(stride) == (1, 1)
                     else nn.Conv2d(c_in, c_out, 1, stride=stride))

    def forward(self, h):
        r = self.proj(h)
        h = self.drop(F.gelu(self.norm1(self.conv1(h))))
        h = self.drop(F.gelu(self.norm2(self.conv2(h))))
        return h + r


class VibrationConv2dEncoder(nn.Module):
    """DSP frontend -> 2D conv trunk -> channel identity -> flat token sequence."""

    def __init__(self, d_model=128,
                 n_channels=8,
                 fs=25600.0,
                 snapshot_len=25600,
                 n_fft=5120,
                 hop=1280,
                 n_freq=64,
                 env_band=(2000.0, 10000.0),
                 env_fmax=600.0,
                 stem_channels=32,
                 channels=(64, 128, 128, 128),
                 # (freq, time) per stage. The 200 ms analysis window leaves
                 # only 17 frames to begin with, so time is downsampled half as
                 # aggressively as frequency to keep a usable token count.
                 strides=((2, 1), (2, 1), (2, 2), (2, 2)),
                 n_groups=8,
                 dropout=0.1):
        super().__init__()
        assert len(channels) == len(strides), \
            f"{len(channels)} stages but {len(strides)} strides"
        assert channels[-1] == d_model, \
            f"last stage width {channels[-1]} must equal d_model {d_model}"
        assert stem_channels % n_groups == 0, \
            f"stem {stem_channels} not divisible by {n_groups} groups"

        self.d_model = d_model
        self.n_channels = n_channels
        self.dsp = VibrationDSP(fs, snapshot_len, n_fft, hop, n_freq,
                                env_band, env_fmax)

        self.stem = nn.Sequential(
            nn.Conv2d(2, stem_channels, 3, padding=1),
            nn.GroupNorm(n_groups, stem_channels),
            nn.GELU(),
        )
        widths = (stem_channels,) + tuple(channels)
        self.blocks = nn.ModuleList([
            ConvBlock2d(widths[i], widths[i + 1], strides[i], n_groups, dropout)
            for i in range(len(channels))
        ])

        # walk the strides so the token count is known before any data arrives
        f, t = n_freq, self.dsp.n_frames
        for sf, st in strides:
            f, t = (f - 1) // sf + 1, (t - 1) // st + 1
        self.out_freq, self.out_len = f, t
        assert f >= 1 and t >= 1, \
            f"strides collapse the grid to ({f}, {t}); use fewer stages"

        self.channel_embed = nn.Embedding(n_channels, d_model)
        # Per-snapshot RMS normalization throws away absolute amplitude, which
        # is what makes the encoder portable across sensor sensitivities and
        # turbine sizes. But broadband level is itself a health indicator
        # (ISO 10816), so it comes back as its own token rather than being
        # lost. The mask rides along so the model can tell a quiet sensor from
        # a dead one -- both give an all-zero spectrogram otherwise.
        self.level_proj = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, x, mask=None):
        B, C, L = x.shape
        assert C == self.n_channels, \
            f"got {C} accelerometers, built for {self.n_channels}"
        if mask is None:
            mask = x.new_ones(B, C)
        mf = mask.reshape(B * C)

        img, level = self.dsp(x.reshape(B * C, L), mf)

        h = self.stem(img)
        for blk in self.blocks:
            h = blk(h)                       # (B*C, d_model, out_freq, out_len)

        # Pool the frequency axis away and keep time as the token axis. Band
        # identity is not lost: by the last stage each filter's receptive field
        # spans most of the frequency extent, so which bands are hot is encoded
        # in the d_model features rather than in a surviving spatial axis.
        h = h.mean(dim=2)                    # (B*C, d_model, out_len)
        h = h.transpose(1, 2)                # (B*C, out_len, d_model)
        h = h.reshape(B, C, self.out_len, self.d_model)

        ch = torch.arange(C, device=x.device)
        emb = self.channel_embed(ch)                     # (C, d_model)
        h = h + emb[None, :, None, :]

        # channel-major, matching ScadaTCNEncoder: channel c owns [c*P, (c+1)*P)
        seq = h.reshape(B, C * self.out_len, self.d_model)

        lvl = self.level_proj(torch.cat([level, mf[:, None]], dim=-1))
        lvl = lvl.reshape(B, C, self.d_model) + emb[None, :, :]

        return torch.cat([seq, lvl], dim=1)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, L = 4, 8, 25600

    enc = VibrationConv2dEncoder().eval()
    x = torch.randn(B, C, L) * 0.5
    m = torch.ones(B, C)
    m[0, 3] = 0.0                                   # one dead accelerometer

    with torch.no_grad():
        img, level = enc.dsp(x.reshape(B * C, L), m.reshape(B * C))
        tokens = enc(x, m)

    n = sum(p.numel() for p in enc.parameters())
    print(f"dsp image      {tuple(img.shape)}   (planes: raw, envelope)")
    print(f"grid           {enc.dsp.n_freq} freq x {enc.dsp.n_frames} frames"
          f"  ->  {enc.out_freq} x {enc.out_len} after the trunk")
    print(f"tokens         {tuple(tokens.shape)}"
          f"   = {C}x{enc.out_len} spectro + {C} level")
    print(f"finite         {bool(torch.isfinite(tokens).all())}")
    print(f"parameters     {n:,}  ({n * 4 / 1e6:.1f} MB fp32)")
