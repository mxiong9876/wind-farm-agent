"""Dataset loaders: real sensor archives -> tensors the models accept.

NOT named `io`. A package called `io` at the repo root shadows Python's stdlib
`io` module for anything that puts the root on sys.path -- which the test
bootstraps all do -- and pandas, torch and numpy all import it. The failure is
loud but deeply confusing, so the name is deliberate.
"""
