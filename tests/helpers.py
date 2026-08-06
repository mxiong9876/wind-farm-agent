"""Shared helpers for the stage tests.

Every stage test imports from here so the causality check and the pass/fail
reporting stay identical across files.
"""

import torch

TOL = 1e-4          # float32 noise sits around 1e-6; a real leak is order 1.0


def banner(name):
    print(f"\n=== {name} ===")


def report(label, value, ok, fmt="{:.2e}"):
    status = "PASS" if ok else "FAIL"
    print(f"  {label:<38} {fmt.format(value):>12}  {status}")
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
    print(f"  {name:<38} {str(got):>12}  {status}")
    assert ok, f"{name}: got {got}, expected {tuple(expected)}"
