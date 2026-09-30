"""Walk-forward backtest on real NYT answers: each day only uses answers published before that day.

Variants (all use the same one-step guess rule; only the candidate prior and the opener differ):
  T0   textbook: uniform over the original 2,315-word list, SALET opener (the "mathematical optimum" setting)
  T1   textbook minus words already used as answers, SALET
  P0   the tool's base prior over all 12,972 words (no history), TARES
  P2   history-aware prior: three categories (unused original / already used / outside the original list),
       category shares estimated from the previous WINDOW days, base prior inside each category, TARES
  P2s  P2 with the SALET opener (what the web tool ships)

Needs data/nyt_history.json (fetch_history.py). Days after LAST_DAY (env, default 2026-09-25) are ignored so
the numbers in EXPERIMENTS.md reproduce exactly.
usage: python3 experiments/backtest.py T0,T1,P0,P2,P2s
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from common import DATA, IS_ANS, INDEX, N, Q, choose, save_json, tool_prior

LAST_DAY = os.environ.get("LAST_DAY", "2026-09-25")
H = json.load(open(os.path.join(DATA, "nyt_history.json")))
DAYS = [d for d in sorted(H) if d <= LAST_DAY]
SOL = {d: INDEX[H[d]["solution"]] for d in DAYS}
ORIG = IS_ANS
PRIOR = tool_prior()
WINDOW = 180
PERIODS = {"2025": [d for d in DAYS if d.startswith("2025")], "2026": [d for d in DAYS if d.startswith("2026")]}


def used_before(day):
    m = np.zeros(N, bool)
    for d in DAYS:
        if d >= day:
            break
        m[SOL[d]] = True
    return m


def category_rates(day):
    """Share of recent answers that were unused-original / repeats / outside the original list."""
    prior_days = [d for d in DAYS if d < day][-WINDOW:]
    seen = used_before(prior_days[0]) if prior_days else np.zeros(N, bool)
    c = np.ones(3)  # add-one smoothing
    for d in prior_days:
        a = SOL[d]
        c[1 if seen[a] else 0 if ORIG[a] else 2] += 1
        seen[a] = True
    return c / c.sum()


def weights(variant, day):
    used = used_before(day)
    if variant == "T0":
        return ORIG.astype(float)
    if variant == "T1":
        return (ORIG & ~used).astype(float)
    if variant == "P0":
        return PRIOR
    if variant.startswith("P2"):
        r = category_rates(day)
        cats = [ORIG & ~used, used, ~ORIG & ~used]
        w = np.zeros(N)
        for share, m in zip(r, cats):
            if m.any():
                w[m] = share * PRIOR[m] / PRIOR[m].sum()
        return w
    raise ValueError(variant)


OPENER = {"T0": "salet", "T1": "salet", "P0": "tares", "P2": "tares", "P2s": "salet"}


def play_with(w, opener, ans, c=(0.55, 0.1)):
    """Guesses needed for answer index ans; 99 if the answer was never a candidate. c: cost constants."""
    S = np.flatnonzero(w > 0)
    g = INDEX[opener]
    for turn in range(1, 9):
        if g == ans:
            return turn
        S = S[Q[g, S] == Q[g, ans]]
        if len(S) == 0:
            return 99
        g = choose(S, w, *c)
    return 9


def play(args):
    variant, day = args
    return variant, day, play_with(weights(variant, day), OPENER[variant], SOL[day])


def paired(out, a, b, days):
    both = [d for d in days if out[a][d] < 99 and out[b][d] < 99]
    diff = np.array([out[a][d] - out[b][d] for d in both], float)
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    return {"a": a, "b": b, "days": len(both), "diff": diff.mean(), "ci95": [diff.mean() - 1.96 * se, diff.mean() + 1.96 * se],
            "fewer_same_more": [int((diff < 0).sum()), int((diff == 0).sum()), int((diff > 0).sum())]}


def report(out, variants, ref="T0"):
    summary = {"periods": {}, "paired": []}
    for v in variants:
        for name, days in list(PERIODS.items()) + [("all", PERIODS["2025"] + PERIODS["2026"])]:
            n = np.array([out[v][d] for d in days])
            ok = n < 99
            s = {"days": len(days), "avg_found": n[ok].mean(), "not_found": int((~ok).sum()),
                 "over6": int((ok & (n > 6)).sum()), "le3": (n <= 3).mean()}
            summary["periods"].setdefault(v, {})[name] = s
            print(f"{v:5} {name:4}  {len(days)} days  avg {s['avg_found']:.3f} (days the answer was findable)  "
                  f"not found {s['not_found']}  over 6: {s['over6']}  solved in <=3: {s['le3']*100:.1f}%", flush=True)
    for v in variants:
        if v == ref or ref not in variants:
            continue
        for name, days in PERIODS.items():
            p = paired(out, v, ref, days)
            summary["paired"].append(dict(p, period=name))
            print(f"{name} {v} - {ref} on {p['days']} days both could solve: {p['diff']:+.3f}  "
                  f"95% CI [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]  fewer/same/more {p['fewer_same_more']}")
    return summary


if __name__ == "__main__":
    variants = (sys.argv[1] if len(sys.argv) > 1 else "T0,T1,P0,P2,P2s").split(",")
    tasks = [(v, d) for v in variants for p in PERIODS.values() for d in p]
    t = time.time()
    with Pool(4) as pool:
        res = pool.map(play, tasks, chunksize=8)
    out = {}
    for v, d, n in res:
        out.setdefault(v, {})[d] = n
    summary = report(out, variants)
    save_json("backtest_days.json", out)
    save_json("backtest_summary.json", summary)
    print(f"{time.time()-t:.0f}s")
