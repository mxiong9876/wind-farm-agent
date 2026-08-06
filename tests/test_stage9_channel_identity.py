"""Stage 9 -- channel identity: tell fusion which sensor each token came from."""

import torch
from scada_encoder_tcn import ScadaTCNEncoder
from helpers import banner, report


def test():
    banner("stage 9: channel identity")
    torch.manual_seed(0)
    B, C, T, d = 4, 20, 600, 128

    enc = ScadaTCNEncoder(d_model=d, n_channels=C, context_len=T).eval()

    # identical DATA on two channels must still yield different tokens,
    # because the channel embedding distinguishes them
    x = torch.randn(B, C, T)
    x[:, 1] = x[:, 0]
    with torch.no_grad():
        out = enc(x, torch.ones(B, C, T))
    diff = (out[:, 0] - out[:, 1]).abs().max().item()
    report("identical data, channels differ", diff, diff > 1e-3, fmt="{:.4f}")

    # the embedding must be a constant per channel, so the difference between
    # two channels fed identical data is the same at every token position
    with torch.no_grad():
        delta = out[:, 0] - out[:, 1]
    spread = (delta - delta.mean(dim=1, keepdim=True)).abs().max().item()
    report("offset constant across tokens", spread, spread < 1e-3)

    # embedding must be learnable
    emb = enc.channel_embed
    trainable = emb.weight.requires_grad
    report("embedding is trainable", float(trainable), trainable, fmt="{:.0f}")
    report("embedding rows == n_channels", emb.weight.shape[0],
           emb.weight.shape[0] == C, fmt="{:.0f}")


if __name__ == "__main__":
    test()
