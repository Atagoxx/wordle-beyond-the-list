"""Exploratory, fixed-date pilot of two alternative continuation-cost approximations.

All variants use the same P2 history prior and SALET opener. No fitting is done.
The sample and paired intervals describe this historical pilot, not a new holdout.

usage: python3 experiments/pilot_branch_cost.py --limit 96 --workers 2
"""
import argparse
import hashlib
import json
import os
import random
import time
from multiprocessing import Pool

os.environ["LAST_DAY"] = "2026-09-25"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

import backtest as bt
from common import ALL_GREEN, G, INDEX, N, Q, RESULTS, choose

VARIANTS = ("baseline", "branch", "conditional")
MAX_TURNS = 12


def choose_alternative(S, w, variant, c1=0.55, c2=0.1):
    """Return (alternative action, original-rule action) at this same state.

    branch: 1 + sum_{b != green} P(b) * phi(H(answer | b)).
    conditional: 1 + P(fail) * phi(H(answer | feedback) / P(fail)).
    phi(h) = 1 + c1*h + c2*h**2. Neither expression is an exact value function.
    """
    if len(S) <= 2:
        g = choose(S, w, c1, c2)
        return int(g), int(g)
    if variant not in ("branch", "conditional"):
        raise ValueError(variant)

    p = w[S] / w[S].sum()
    plogp = np.zeros_like(p)
    positive = p > 0
    plogp[positive] = p[positive] * np.log2(p[positive])
    hs = -plogp.sum()
    pwin = np.zeros(N)
    pwin[S] = p
    base_cost = np.empty(N)
    alt_cost = np.empty(N)

    for start in range(0, N, 1500):
        stop = min(start + 1500, N)
        nrows = stop - start
        bins = Q[start:stop][:, S].astype(np.int64)
        bins += (np.arange(nrows) * 243)[:, None]
        mass = np.bincount(
            bins.ravel(), weights=np.broadcast_to(p, bins.shape).ravel(),
            minlength=nrows * 243,
        ).reshape(nrows, 243)
        logmass = np.zeros_like(mass)
        np.log2(mass, out=logmass, where=mass > 0)
        hg = -(mass * logmass).sum(axis=1)
        rem = np.maximum(hs - hg, 0)
        fail = 1 - pwin[start:stop]
        base_cost[start:stop] = 1 + fail * (1 + c1 * rem + c2 * rem ** 2)

        if variant == "conditional":
            hfail = np.divide(rem, fail, out=np.zeros_like(rem), where=fail > 0)
            alt_cost[start:stop] = 1 + fail * (1 + c1 * hfail + c2 * hfail ** 2)
        else:
            sum_plogp = np.bincount(
                bins.ravel(), weights=np.broadcast_to(plogp, bins.shape).ravel(),
                minlength=nrows * 243,
            ).reshape(nrows, 243)
            hb = logmass - np.divide(
                sum_plogp, mass, out=np.zeros_like(mass), where=mass > 0,
            )
            hb = np.maximum(hb, 0)
            continuation = mass * (1 + c1 * hb + c2 * hb ** 2)
            continuation[:, ALL_GREEN] = 0
            alt_cost[start:stop] = 1 + continuation.sum(axis=1)
    return int(np.argmin(alt_cost)), int(np.argmin(base_cost))


def play(task):
    variant, day = task
    w = bt.weights("P2s", day)
    answer = bt.SOL[day]
    S = np.flatnonzero(w > 0)
    g = INDEX["salet"]
    guesses = []
    changes = []
    compared = 0
    for turn in range(1, MAX_TURNS + 1):
        guesses.append(G[g])
        if g == answer:
            return variant, day, {
                "turns": turn, "solved": True, "not_found": False, "capped": False,
                "guesses": guesses, "decisions_compared": compared,
                "decision_changes": len(changes), "changed_decisions": changes,
            }
        S = S[Q[g, S] == Q[g, answer]]
        if not len(S):
            return variant, day, {
                "turns": 99, "solved": False, "not_found": True, "capped": False,
                "guesses": guesses, "decisions_compared": compared,
                "decision_changes": len(changes), "changed_decisions": changes,
            }
        if turn == MAX_TURNS:
            break
        if variant == "baseline":
            g = int(choose(S, w))
        else:
            g, baseline_g = choose_alternative(S, w, variant)
            compared += 1
            if g != baseline_g:
                changes.append({
                    "turn": turn + 1, "candidates": len(S),
                    "selected": G[g], "baseline_at_same_state": G[baseline_g],
                })
    return variant, day, {
        "turns": MAX_TURNS + 1, "solved": False, "not_found": False, "capped": True,
        "guesses": guesses, "decisions_compared": compared,
        "decision_changes": len(changes), "changed_decisions": changes,
    }


def summarize(per_day, days):
    summaries = {}
    for variant in VARIANTS:
        rows = [per_day[variant][d] for d in days]
        solved = [r["turns"] for r in rows if r["solved"]]
        dist = {str(n): solved.count(n) for n in sorted(set(solved))}
        summaries[variant] = {
            "days": len(days), "solved": len(solved),
            "avg_solved": float(np.mean(solved)) if solved else None,
            "total_guesses_solved": sum(solved), "distribution_solved": dist,
            "failed_within_6": sum(r["turns"] > 6 for r in rows),
            "not_found": sum(r["not_found"] for r in rows),
            "capped_at_12": sum(r["capped"] for r in rows),
            "solved_le3_fraction": sum(r["solved"] and r["turns"] <= 3 for r in rows) / len(rows),
            "decisions_compared": sum(r["decisions_compared"] for r in rows),
            "decision_changes": sum(r["decision_changes"] for r in rows),
            "games_with_decision_changes": sum(r["decision_changes"] > 0 for r in rows),
        }
    paired = {}
    for variant in VARIANTS[1:]:
        both = [d for d in days if per_day[variant][d]["solved"] and per_day["baseline"][d]["solved"]]
        diff = np.array([
            per_day[variant][d]["turns"] - per_day["baseline"][d]["turns"] for d in both
        ], float)
        mean = float(diff.mean()) if len(diff) else None
        halfwidth = float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else None
        paired[variant] = {
            "reference": "baseline", "both_solved_days": len(both),
            "mean_turn_difference": mean,
            "descriptive_normal_ci95": [mean - halfwidth, mean + halfwidth] if halfwidth is not None else None,
            "net_turns_saved": int(-diff.sum()),
            "fewer_same_more": [int((diff < 0).sum()), int((diff == 0).sum()), int((diff > 0).sum())],
        }
    return summaries, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=96, help="Uniform sampled days; 0 means all 633 days.")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", help="JSON path; default results/pilot_branch_cost.json")
    parser.add_argument("--source-label", default="Local data/nyt_history.json; provenance to be verified",
                        help="Describe the historical-answer source used for this run.")
    args = parser.parse_args()
    if args.limit < 0 or args.workers < 1:
        parser.error("--limit must be nonnegative and --workers must be positive")
    population = [d for d in bt.DAYS if "2025-01-01" <= d <= "2026-09-25"]
    if not population:
        raise RuntimeError("No evaluation dates in the required history window")
    limit = min(args.limit or len(population), len(population))
    days = sorted(random.Random(args.seed).sample(population, limit))
    tasks = [(v, d) for v in VARIANTS for d in days]
    per_day = {v: {} for v in VARIANTS}
    t0 = time.time()
    print(f"Historical pilot: {len(days)}/{len(population)} uniformly sampled days, seed {args.seed}; "
          f"{args.workers} workers, three variants, P2s prior, SALET, no coefficient fitting.", flush=True)
    with Pool(args.workers) as pool:
        for completed, (variant, day, row) in enumerate(pool.imap_unordered(play, tasks, chunksize=2), 1):
            per_day[variant][day] = row
            count = len(per_day[variant])
            if count % 12 == 0 or count == len(days):
                current = list(per_day[variant].values())
                solved = [r["turns"] for r in current if r["solved"]]
                avg = float(np.mean(solved)) if solved else float("nan")
                print(f"{variant:11} {count:3}/{len(days)} days; partial avg {avg:.4f}; "
                      f"{completed}/{len(tasks)} tasks; {time.time()-t0:.1f}s", flush=True)
    summaries, paired = summarize(per_day, days)
    reference_path = os.path.join(RESULTS, "backtest_days.json")
    reference_check = {"path": reference_path, "available": os.path.exists(reference_path)}
    if reference_check["available"]:
        with open(reference_path) as f:
            saved = json.load(f)["P2s"]
        checked = [d for d in days if d in saved]
        mismatches = [
            {"day": d, "saved": saved[d], "rerun": per_day["baseline"][d]["turns"]}
            for d in checked if saved[d] != per_day["baseline"][d]["turns"]
        ]
        reference_check.update(checked_days=len(checked), mismatch_count=len(mismatches), mismatches=mismatches)
    result = {
        "exploratory": True,
        "caution": "Same historical data already inspected; descriptive paired intervals, not an independent holdout claim.",
        "date_start": "2025-01-01", "date_end": "2026-09-25",
        "population_days": len(population), "sample_days": len(days), "seed": args.seed,
        "history_source": args.source_label,
        "history_sha256": hashlib.sha256(
            open(os.path.join(bt.DATA, "nyt_history.json"), "rb").read()
        ).hexdigest(),
        "sampling": "Python random.Random(seed).sample uniformly without replacement; then chronological sorting",
        "prior": "P2s history-aware", "opener": "salet", "coefficients": [0.55, 0.1],
        "max_turns": MAX_TURNS, "days": days, "summary": summaries, "paired": paired,
        "reference_baseline": reference_check,
        "decision_changes_definition": "Alternative versus original-rule action at each state actually reached by the alternative; excludes fixed opener.",
        "per_day": per_day, "elapsed_seconds": round(time.time() - t0, 3),
    }
    output = args.output or os.path.join(RESULTS, "pilot_branch_cost.json")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    for variant in VARIANTS:
        print(variant, json.dumps(summaries[variant]), flush=True)
    print("paired", json.dumps(paired), flush=True)
    print("reference_baseline", json.dumps(reference_check), flush=True)
    print(f"Wrote {os.path.abspath(output)} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
