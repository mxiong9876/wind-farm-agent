"""Shared helpers for the stage tests.

Every stage test imports from here so the causality check and the pass/fail
reporting stay identical across files.
"""

import sys

import torch

# Line-buffer stdout for the whole process. The explicit flush=True below only
# covers prints that route through these helpers; every suite also prints
# context lines directly ("cached 384 train / 192 test", "windows 11105"), and
# those are exactly the lines that explain a failure. Reconfiguring the stream
# catches them too, so no future print has to remember.
#
# Guarded because stdout is not always a TextIOWrapper -- under some capture
# harnesses it is a plain object with no reconfigure, and a test suite must not
# fail on the way it is being watched.
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

TOL = 1e-4          # float32 noise sits around 1e-6; a real leak is order 1.0

# Every print here flushes. Python line-buffers stdout to a terminal but
# BLOCK-buffers it to a pipe or a file, so `run_all.py | tee log` holds output
# in a 4-8KB buffer until it fills. A suite killed before that -- a laptop
# closing, a session ending, an out-of-memory kill -- loses everything it had
# already reported, and the surviving evidence is an empty log rather than the
# assertions that passed. Measured twice on this project: a training run and
# the vlm suite both produced no output at all when interrupted while piped.
#
# The vlm suite is the reason this is not merely tidy. It loads a multi-GB
# checkpoint, so it is the slowest suite and the likeliest to be interrupted,
# and without flushing an interrupted run is indistinguishable from one that
# never started.


def banner(name):
    print(f"\n=== {name} ===", flush=True)


def report(label, value, ok, fmt="{:.2e}"):
    status = "PASS" if ok else "FAIL"
    print(f"  {label:<38} {fmt.format(value):>12}  {status}", flush=True)
    assert ok, f"{label}: {value}"


def check_causal(module, N=80, d=128, T=600, cut=400, tol=TOL, label="causal"):
    """For modules with signature (h) -> h, where h is (N, d_model, T).

    Perturbs everything at or after `cut`, then confirms outputs BEFORE `cut`
    are unchanged. A causal module cannot see the future, so the difference
    must be zero up to float32 noise.
    """
    module.eval()
    h = torch.randn(N, d, T)
    h2 = h.clone()
    h2[:, :, cut:] += 10.0
    with torch.no_grad():
        a, b = module(h), module(h2)
    leak = (a[:, :, :cut] - b[:, :, :cut]).abs().max().item()
    report(label, leak, leak < tol)
    return leak


def check_trunk_causal(trunk, N=80, T=600, cut=400, tol=TOL):
    """Same idea, but TCNTrunk's signature is (x, mask) rather than (h).

    NOTE: RevIN is deliberately excluded from this check. RevIN normalizes by
    whole-window statistics, so changing t=500 shifts the mean and std applied
    at t=100 -- it is non-causal BY DESIGN. That is correct for the health
    path, which analyses a fixed historical window. If this encoder is ever
    reused for the control path, RevIN must be swapped for a running-statistics
    variant, and only then should causality be asserted end to end.
    """
    trunk.eval()
    x = torch.randn(N, T)
    m = torch.ones(N, T)
    x2 = x.clone()
    x2[:, cut:] += 10.0
    with torch.no_grad():
        a, b = trunk(x, m), trunk(x2, m)
    leak = (a[:, :, :cut] - b[:, :, :cut]).abs().max().item()
    report("trunk causal (RevIN excluded)", leak, leak < tol)
    return leak


def check_shape(name, tensor, expected):
    got = tuple(tensor.shape)
    ok = got == tuple(expected)
    status = "PASS" if ok else "FAIL"
    print(f"  {name:<38} {str(got):>12}  {status}", flush=True)
    assert ok, f"{name}: got {got}, expected {tuple(expected)}"
