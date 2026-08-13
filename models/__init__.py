"""Encoders and the fusion body.

Each module here is a nn.Module and nothing else -- no data loading, no
training loop, no file paths. That separation is what lets the same encoder be
exercised by a synthetic smoke test and by real Kelmarsh windows without
either knowing about the other.
"""
