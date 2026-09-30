"""Choose the opener for the history-aware prior WITHOUT looking at the answers, then check against reality.

Step 1 (no answers used): for each opener, the expected number of guesses under the P2 prior of a fixed day,
computed exactly by walking the solver's full decision tree over every candidate, weighted by its prior mass.
Step 2 (the real answers): walk-forward backtest of P2 with that opener on 2025-01-01 .. 2026-09-25.

usage: python3 experiments/opener_history.py [day] [openers]   (defaults: 2025-01-01, 12 common openers)
"""
import sys
import time
from multiprocessing import Pool

import numpy as np

import backtest as bt
from common import ALL_GREEN, INDEX, Q, choose, save_json

DAY = sys.argv[1] if len(sys.argv) > 1 else "2025-01-01"
OPENERS = (sys.argv[2] if len(sys.argv) > 2 else
           "salet,tares,reast,slate,crane,trace,crate,soare,roate,raise,rates,lares").split(",")


def expected_under_prior(opener):
    w = bt.weights("P2", DAY)
    p = w / w.sum()

    def walk(S, depth, forced=None):
        g = forced if forced is not None else choose(S, w)
        codes = Q[g, S]
        tot = 0.0
        for code in np.unique(codes):
            if code == ALL_GREEN:
                tot += p[g] * depth
            else:
                tot += walk(S[codes == code], depth + 1)
        return tot

    return opener, walk(np.flatnonzero(w > 0), 1, INDEX[opener])


def play(args):
    opener, day = args
    return opener, day, bt.play_with(bt.weights("P2", day), opener, bt.SOL[day])


if __name__ == "__main__":
    t = time.time()
    with Pool(4) as pool:
        model = dict(pool.map(expected_under_prior, OPENERS))
        days = bt.PERIODS["2025"] + bt.PERIODS["2026"]
        res = pool.map(play, [(o, d) for o in OPENERS for d in days], chunksize=8)
    real = {}
    for o, d, n in res:
        real.setdefault(o, {})[d] = n
    rows = []
    for o in sorted(OPENERS, key=lambda o: model[o]):
        r = np.array([real[o][d] for d in days])
        rows.append({"opener": o, "expected_under_prior": model[o], "backtest_avg": r.mean(),
                     "backtest_2025": np.mean([real[o][d] for d in bt.PERIODS["2025"]]),
                     "backtest_2026": np.mean([real[o][d] for d in bt.PERIODS["2026"]]),
                     "over6": int((r > 6).sum())})
        x = rows[-1]
        print(f"{o.upper():6} expected under the {DAY} prior {x['expected_under_prior']:.4f}  | real answers: "
              f"all {x['backtest_avg']:.3f}  2025 {x['backtest_2025']:.3f}  2026 {x['backtest_2026']:.3f}  "
              f"over 6: {x['over6']}", flush=True)
    rho = np.corrcoef(np.argsort(np.argsort([x["expected_under_prior"] for x in rows])),
                      np.argsort(np.argsort([x["backtest_avg"] for x in rows])))[0, 1]
    print(f"rank correlation, prior estimate vs real backtest: {rho:.2f}")
    save_json("opener_history.json", {"day": DAY, "rows": rows, "rank_corr": rho, "per_day": real})
    print(f"{time.time()-t:.0f}s")
