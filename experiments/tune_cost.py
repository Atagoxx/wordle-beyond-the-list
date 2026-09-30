"""Tune the two hand-set constants of the guess rule, cost = 1 + (1 - P(win)) * (1 + c1*rem + c2*rem^2),
on 2025 only (history-aware prior P2, SALET opener), then check the chosen pair once on 2026.

usage: python3 experiments/tune_cost.py
"""
import itertools
import time
from multiprocessing import Pool

import numpy as np

import backtest as bt
from common import save_json

DEFAULT = (0.55, 0.1)
GRID = list(itertools.product([0.35, 0.45, 0.55, 0.65, 0.75], [0.05, 0.1, 0.15, 0.2]))


def play(args):
    c, day = args
    return c, day, bt.play_with(bt.weights("P2", day), "salet", bt.SOL[day], c)


def run(cs, days):
    with Pool(4) as pool:
        res = pool.map(play, [(c, d) for c in cs for d in days], chunksize=8)
    out = {}
    for c, d, n in res:
        out.setdefault(c, {})[d] = n
    return out


if __name__ == "__main__":
    t = time.time()
    dev, test = bt.PERIODS["2025"], bt.PERIODS["2026"]
    dev_out = run(GRID, dev)
    rows = sorted(((np.mean([dev_out[c][d] for d in dev]), c) for c in GRID))
    print("2025 (tuning) average guesses, best first:")
    for avg, c in rows:
        print(f"  c1={c[0]:.2f} c2={c[1]:.2f}  {avg:.3f}{'   <- default' if c == DEFAULT else ''}")
    best = rows[0][1]
    test_out = run([DEFAULT, best] if best != DEFAULT else [DEFAULT], test)
    a = np.array([test_out[DEFAULT][d] for d in test])
    b = np.array([test_out[best][d] for d in test])
    diff = b - a
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if diff.any() else 0.0
    print(f"2026 (held out): default {a.mean():.3f}  tuned c1={best[0]} c2={best[1]} {b.mean():.3f}  "
          f"diff {diff.mean():+.3f} (95% CI +/- {1.96*se:.3f}), over 6: {(a > 6).sum()} vs {(b > 6).sum()}")
    save_json("tune_cost.json", {"dev_2025": [{"c1": c[0], "c2": c[1], "avg": avg} for avg, c in rows],
                                 "chosen": best, "test_2026": {"default": a.mean(), "tuned": b.mean(),
                                                               "diff": diff.mean(), "ci95_halfwidth": 1.96 * se}})
    print(f"{time.time()-t:.0f}s")
