"""Train the fusion -> Qwen3-VL projector on manufactured (health, text) pairs.

    python3 scripts/build_vlm_pairs.py           # first: make the pairs
    python3 scripts/train_vlm_projector.py       # then: train the adapter

    python3 scripts/train_vlm_projector.py --epochs 1 --limit 32   # smoke run
    python3 scripts/train_vlm_projector.py --model Qwen/Qwen3-VL-8B-Instruct

WHAT TRAINS

Only the projector -- 4.5M parameters against a frozen 2.1B. Gradient still
flows backward THROUGH the language model to reach it, so the memory cost is a
full backward pass even though almost nothing is updated. That is what decides
where this can run: the 2B fits on a laptop, the 8B does not.

THE METRIC THAT MATTERS IS NOT THE LOSS

Loss falls even when the projector is ignored, because the brief alone predicts
much of the target. The check that means something is the ABLATION reported
each epoch: the same batch scored with the health vectors shuffled between
samples. If shuffling does not hurt, the model is reading the brief and routing
around the soft tokens, and the adapter is decorative no matter how good the
loss looks.

    val loss 1.82   shuffled 1.84   -> ignoring the health vector
    val loss 1.82   shuffled 3.10   -> genuinely conditioned on it

build_vlm_pairs.py is written to make that gap possible: the residual appears
in the target and never in the brief, so it is reachable only through the
health vector.

OUTPUT

    runs/vlm_projector/best.pt        projector weights only (~18MB)
    runs/vlm_projector/history.csv    per-epoch train/val/shuffled loss
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.vlm_bridge import FusionToVLM


def load_pairs(d, limit=0):
    with open(os.path.join(d, "pairs.jsonl")) as f:
        rows = [json.loads(l) for l in f]
    health = torch.load(os.path.join(d, "health.pt"), map_location="cpu")
    if limit:
        rows, health = rows[:limit], health[:limit]
    tr = [i for i, r in enumerate(rows) if r["split"] == "train"]
    va = [i for i, r in enumerate(rows) if r["split"] == "val"]
    return rows, health, tr, va


def run_batch(bridge, rows, health, idx, shuffle_health=False):
    """One forward. `shuffle_health` is the ablation, not a bug."""
    h = health[idx]
    if shuffle_health:
        # break the pairing while keeping the marginal distribution identical,
        # so any loss increase is attributable to the CORRESPONDENCE between a
        # health vector and its text, not to the vectors being unusual
        h = h[torch.randperm(len(h))]
    return bridge(health=h,
                  text=[rows[i]["brief"] for i in idx],
                  labels_text=[rows[i]["target"] for i in idx])


@torch.no_grad()
def evaluate(bridge, rows, health, ids, batch, shuffle_health=False):
    bridge.eval()
    tot, n = 0.0, 0
    for i in range(0, len(ids), batch):
        b = torch.tensor(ids[i:i + batch])
        if len(b) < 2:
            continue
        tot += run_batch(bridge, rows, health, b, shuffle_health).loss.item()
        n += 1
    return tot / max(n, 1)


def main(a):
    rows, health, tr, va = load_pairs(a.pairs, a.limit)
    print(f"pairs: {len(rows)}   train {len(tr)}   val {len(va)}")
    if not tr or not va:
        raise SystemExit("need both train and val rows; check --val-year in "
                         "build_vlm_pairs.py")

    print(f"loading {a.model} ...")
    t0 = time.time()
    bridge = FusionToVLM(a.model, d_fusion=health.shape[1], n_soft=a.n_soft)
    dev = next(bridge.vlm.parameters()).device
    n_tr = sum(p.numel() for p in bridge.trainable_parameters())
    print(f"  {time.time()-t0:.0f}s   device {dev}   d_lm {bridge.d_lm}")
    print(f"  trainable {n_tr:,}  of  "
          f"{sum(p.numel() for p in bridge.parameters()):,}\n")

    opt = torch.optim.AdamW(bridge.trainable_parameters(), lr=a.lr,
                            weight_decay=a.weight_decay)
    os.makedirs(a.out, exist_ok=True)
    hist, best = [], float("inf")

    for ep in range(a.epochs):
        bridge.train()
        perm = torch.randperm(len(tr))
        tot, n, t0 = 0.0, 0, time.time()
        for i in range(0, len(tr) - a.batch + 1, a.batch):
            idx = torch.tensor([tr[j] for j in perm[i:i + a.batch]])
            loss = run_batch(bridge, rows, health, idx).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bridge.trainable_parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1

        train_loss = tot / max(n, 1)
        val = evaluate(bridge, rows, health, va, a.batch)
        # the ablation: same text, health vectors shuffled between samples
        shuf = evaluate(bridge, rows, health, va, a.batch, shuffle_health=True)

        flag = ""
        if val < best:
            best, flag = val, "  *"
            torch.save({"projector": bridge.projector.state_dict(),
                        "model_id": a.model, "n_soft": a.n_soft,
                        "d_fusion": health.shape[1], "epoch": ep,
                        "val_loss": val, "shuffled_loss": shuf},
                       os.path.join(a.out, "best.pt"))
        hist.append({"epoch": ep, "train": train_loss, "val": val,
                     "shuffled": shuf, "gap": shuf - val})
        with open(os.path.join(a.out, "history.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(hist[0]))
            w.writeheader(); w.writerows(hist)

        print(f"epoch {ep:3d}  train {train_loss:.4f}  val {val:.4f}  "
              f"shuffled {shuf:.4f}  gap {shuf - val:+.4f}  "
              f"{time.time()-t0:.0f}s{flag}", flush=True)

    g = hist[-1]["gap"]
    print(f"\nbest val loss {best:.4f}")
    print(f"final ablation gap {g:+.4f}")
    # a small gap is the honest failure mode here, and it looks like success on
    # the loss curve alone -- say so rather than leaving it to be noticed
    print("  " + ("the projector is conditioning on the health vector"
                  if g > 0.15 else
                  "WARNING: shuffling the health vectors barely hurts. The "
                  "model is reading the brief and ignoring the soft tokens; "
                  "the loss curve is not evidence of a working adapter."))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default="runs/vlm_pairs")
    p.add_argument("--out", default="runs/vlm_projector")
    p.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--n-soft", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
