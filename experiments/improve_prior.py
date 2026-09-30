"""P3: can the history-aware prior (P2) be sharpened without peeking at the future?

P2 puts each category's share on the base prior as is. P3 keeps the same three categories and shares but
fits, walk-forward (only on days before the day being played), how the prior should be shaped inside each one:
  unused original words:  w ∝ base^a_U
  outside the original list: w ∝ base^a_X
  already used (repeats):  w ∝ base^a_R * exp(b * years since the word was first used)
a_* and b are maximum a-posteriori fits on past answers of that category (weak Gaussian priors: a ~ N(1, 1),
b ~ N(0, 1) per year), searched on a grid. With no past answers in a category the defaults a = 1, b = 0 apply,
which is exactly P2.

Disclosure: the repeat-age term was added after seeing that most 2026 repeats were 2021 answers
(history_stats.py). Its value b is still only learned from repeats before each day.

usage: python3 experiments/improve_prior.py [loglik|backtest]   (default: both)
"""
import sys
import time
from multiprocessing import Pool

import numpy as np

import backtest as bt
from common import N, save_json

A_GRID = np.round(np.arange(0.0, 3.01, 0.1), 2)
B_GRID = np.round(np.arange(-1.0, 3.01, 0.1), 2)
LOGB = np.log(bt.PRIOR)
DAY_NUM = {d: i for i, d in enumerate(bt.DAYS)}
FIRST_USE = np.full(N, np.nan)  # day number of each word's first use as an answer
for _d in bt.DAYS:
    if np.isnan(FIRST_USE[bt.SOL[_d]]):
        FIRST_USE[bt.SOL[_d]] = DAY_NUM[_d]


def state(day):
    """Masks of the three categories just before `day`, and the answer's category."""
    used = bt.used_before(day)
    cats = [bt.ORIG & ~used, used, ~bt.ORIG & ~used]
    a = bt.SOL[day]
    return cats, next(i for i, m in enumerate(cats) if m[a])


def logsumexp(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def day_loglik(day):
    """Within-category log-likelihood of the day's answer for every grid value.
    Returns (category, table) with table[a] for U/X and table[a, b] for repeats."""
    cats, c = state(day)
    a = bt.SOL[day]
    m = cats[c]
    lb = LOGB[m]
    if c != 1:
        return c, A_GRID * LOGB[a] - logsumexp(A_GRID[:, None] * lb[None, :])
    age = (DAY_NUM[day] - FIRST_USE[m]) / 365.25
    age_a = (DAY_NUM[day] - FIRST_USE[a]) / 365.25
    z = A_GRID[:, None, None] * lb[None, None, :] + B_GRID[None, :, None] * age[None, None, :]
    return c, A_GRID[:, None] * LOGB[a] + B_GRID[None, :] * age_a - logsumexp(z)


LOGLIK = {}  # day -> (category, table), filled once for every day in the history


def fit(day, window=None):
    """MAP grid fit of a_U, a_R, b, a_X on days strictly before `day` (optionally only the last `window` days)."""
    past = [d for d in bt.DAYS if d < day]
    if window:
        past = past[-window:]
    tot = [np.zeros(len(A_GRID)), np.zeros((len(A_GRID), len(B_GRID))), np.zeros(len(A_GRID))]
    seen = [0, 0, 0]
    for d in past:
        c, t = LOGLIK[d]
        tot[c] += t
        seen[c] += 1
    prior_a = -0.5 * (A_GRID - 1) ** 2
    prior_b = -0.5 * B_GRID ** 2
    out = {}
    for c, key in [(0, "aU"), (2, "aX")]:
        out[key] = float(A_GRID[np.argmax(tot[c] + prior_a)]) if seen[c] else 1.0
    if seen[1]:
        i, j = np.unravel_index(np.argmax(tot[1] + prior_a[:, None] + prior_b[None, :]), tot[1].shape)
        out["aR"], out["b"] = float(A_GRID[i]), float(B_GRID[j])
    else:
        out["aR"], out["b"] = 1.0, 0.0
    return out


def weights_p3(day, params):
    cats, _ = state(day)
    r = bt.category_rates(day)
    w = np.zeros(N)
    for c, (share, m) in enumerate(zip(r, cats)):
        if not m.any():
            continue
        if c == 1:
            age = (DAY_NUM[day] - FIRST_USE[m]) / 365.25
            logits = params["aR"] * LOGB[m] + params["b"] * age
        else:
            logits = params["aX" if c == 2 else "aU"] * LOGB[m]
        e = np.exp(logits - logits.max())
        w[m] = share * e / e.sum()
    return w


def log2p(w, day):
    return float(np.log2(w[bt.SOL[day]] / w.sum()))


def play(args):
    name, day, params = args
    w = bt.weights("P2", day) if name == "P2s" else weights_p3(day, params)
    return name, day, bt.play_with(w, "salet", bt.SOL[day])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    t = time.time()
    for d in bt.DAYS:
        LOGLIK[d] = day_loglik(d)
    test_days = bt.PERIODS["2025"] + bt.PERIODS["2026"]
    variants = {"P3": None, "P3w365": 365}
    params = {v: {d: fit(d, win) for d in test_days} for v, win in variants.items()}
    print(f"fits done ({time.time()-t:.0f}s). Examples of fitted parameters (P3):")
    for d in ["2025-01-01", "2026-01-01", "2026-02-03", "2026-04-01", "2026-06-01", "2026-09-25"]:
        print("  ", d, params["P3"][d])

    # 1) predictive log-likelihood of the real answer (no solver involved): higher = better prior
    ll = {}
    for name in ["P2s"] + list(variants):
        ll[name] = {d: log2p(bt.weights("P2", d) if name == "P2s" else weights_p3(d, params[name][d]), d)
                    for d in test_days}
    print("\nmean log2 P(real answer) under each prior (higher is better)")
    for name in ll:
        row = []
        for per, days in bt.PERIODS.items():
            v = np.array([ll[name][d] for d in days])
            dv = v - np.array([ll["P2s"][d] for d in days])
            row.append(f"{per}: {v.mean():.3f} bits (vs P2 {dv.mean():+.3f} +/- {1.96*dv.std(ddof=1)/np.sqrt(len(dv)):.3f})")
        print(f"  {name:7}", "   ".join(row))
    if mode == "loglik":
        sys.exit()

    # 2) the actual solver, SALET opener, same guess rule
    tasks = [("P2s", d, None) for d in test_days] + [(v, d, params[v][d]) for v in variants for d in test_days]
    with Pool(4) as pool:
        res = pool.map(play, tasks, chunksize=8)
    out = {}
    for name, d, n in res:
        out.setdefault(name, {})[d] = n
    summary = bt.report(out, list(out), ref="P2s")
    save_json("improve_prior.json", {"params": params, "loglik": ll, "per_day": out, "summary": summary})
    print(f"{time.time()-t:.0f}s")
