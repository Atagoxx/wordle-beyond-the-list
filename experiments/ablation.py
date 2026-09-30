"""Ablation: keep the history-aware categories and the SALET opener, swap the within-category word score.
usage: python3 experiments/ablation.py
"""
import time
from multiprocessing import Pool

import numpy as np

import backtest as bt
from common import INDEX, N, Q, choose, load_scores, matched, rank, save_json, tool_prior

BASES = {
    "current: Jev x2 + wordfreq": tool_prior(),
    "Jev only (drop wordfreq)": matched(rank(load_scores("jev_explained_score")) + rank(load_scores("jev_rules_wordle"))),
    "wordfreq only (drop Jev)": matched(load_scores("wordfreq_zipf")),
    "no word score (uniform inside each category)": np.ones(N),
}


def play(args):
    name, day = args
    bt.PRIOR = BASES[name]
    w = bt.weights("P2", day)
    ans = bt.SOL[day]
    S = np.flatnonzero(w > 0)
    g = INDEX["salet"]
    for turn in range(1, 12):
        if g == ans:
            return name, day, turn
        S = S[Q[g, S] == Q[g, ans]]
        g = choose(S, w)
    return name, day, 12


if __name__ == "__main__":
    days = bt.PERIODS["2025"] + bt.PERIODS["2026"]
    t = time.time()
    with Pool(4) as p:
        res = p.map(play, [(n, d) for n in BASES for d in days], chunksize=8)
    out = {}
    for n, d, k in res:
        out.setdefault(n, {})[d] = k
    base = np.array([out["current: Jev x2 + wordfreq"][d] for d in days])
    summary = []
    for n in BASES:
        a = np.array([out[n][d] for d in days])
        diff = a - base
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if diff.any() else 0.0
        summary.append({"base": n, "avg": a.mean(), "over6": int((a > 6).sum()), "le3": (a <= 3).mean(),
                        "diff_vs_current": diff.mean(), "ci95_halfwidth": 1.96 * se})
        print(f"{n:<46} avg {a.mean():.3f}  over 6: {(a > 6).sum()}  solved in <=3: {(a <= 3).mean()*100:.1f}%  "
              f"vs current {diff.mean():+.3f} (+/-{1.96*se:.3f})")
    save_json("ablation.json", {"days": len(days), "summary": summary, "per_day": out})
    print(f"{time.time()-t:.0f}s")
