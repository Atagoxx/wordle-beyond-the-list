"""Exact baseline-policy rollout under each day's P2s prior (exploratory).

SALET remains fixed. From turn two, evaluate up to six candidate actions by
following the original common.choose policy to completion on every feedback
branch. Default mode changes only turn two; --mode all repeats rollout later.
No posterior tail is discarded, and no actual answer enters action selection.

usage: python3 experiments/pilot_rollout.py --limit 96 --workers 2 --mode second
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
from common import ALL_GREEN, DATA, G, INDEX, N, Q, RESULTS, choose

MAX_TURNS = 12
TIE_TOL = 1e-12


class DayPolicy:
    """Caches are private to this day's fixed weights; values are not depth-limited."""

    def __init__(self, weights, top=6):
        self.w = weights
        self.top = top
        self.actions = {}
        self.values = {}
        self.stats = {"baseline_choose_calls": 0, "value_states": 0, "pruned_actions": 0}

    def baseline_action(self, S):
        key = S.tobytes()
        if key not in self.actions:
            self.actions[key] = int(choose(S, self.w))
            self.stats["baseline_choose_calls"] += 1
        return self.actions[key]

    def lower_bound(self, S):
        if len(S) == 1:
            return 1.0
        return float(2 - self.w[S].max() / self.w[S].sum())

    def branches(self, S, g):
        codes = Q[g, S]
        total = self.w[S].sum()
        out = []
        for code in np.unique(codes):
            if code == ALL_GREEN:
                continue
            part = S[codes == code]
            mass = float(self.w[part].sum() / total)
            out.append((mass, part))
        return out

    def value(self, S):
        """Exact expected guesses of the unchanged baseline, including its next guess."""
        key = S.tobytes()
        if key in self.values:
            return self.values[key]
        self.stats["value_states"] += 1
        if len(S) <= 2:
            result = self.lower_bound(S)
        else:
            g = self.baseline_action(S)
            branches = self.branches(S, g)
            if any(len(part) == len(S) for _, part in branches):
                raise RuntimeError("Baseline failed to reduce a nonterminal candidate set")
            result = 1.0 + sum(mass * self.value(part) for mass, part in branches)
        self.values[key] = float(result)
        return float(result)

    def candidate_actions(self, S, baseline):
        """Top heuristic actions, reserving slots for baseline and maximum-posterior word."""
        p = self.w[S] / self.w[S].sum()
        hs = -np.sum(p * np.log2(p))
        pwin = np.zeros(N)
        pwin[S] = p
        costs = np.empty(N)
        for start in range(0, N, 1500):
            stop = min(start + 1500, N)
            bins = Q[start:stop][:, S].astype(np.int64)
            bins += (np.arange(stop - start) * 243)[:, None]
            mass = np.bincount(
                bins.ravel(), weights=np.broadcast_to(p, bins.shape).ravel(),
                minlength=(stop - start) * 243,
            ).reshape(stop - start, 243)
            logmass = np.zeros_like(mass)
            np.log2(mass, out=logmass, where=mass > 0)
            hg = -(mass * logmass).sum(axis=1)
            rem = np.maximum(hs - hg, 0)
            costs[start:stop] = 1 + (1 - pwin[start:stop]) * (1 + 0.55 * rem + 0.1 * rem ** 2)
        order = np.lexsort((np.arange(N), costs))
        required = [baseline, int(S[np.argmax(p)])]
        selected = list(dict.fromkeys(required))
        for g in order:
            if len(selected) >= self.top:
                break
            if int(g) not in selected:
                selected.append(int(g))
        # Evaluate the guaranteed incumbent first, then the others in heuristic order.
        return [baseline] + sorted((g for g in selected if g != baseline), key=lambda g: (costs[g], g))

    def rollout_action(self, S):
        baseline = self.baseline_action(S)
        baseline_value = self.value(S)
        best_g, best_cost = baseline, baseline_value
        evaluations = [{"word": G[baseline], "cost": baseline_value, "exact": True}]
        if len(S) > 2:
            for g in self.candidate_actions(S, baseline)[1:]:
                branches = self.branches(S, g)
                if len(branches) == 1 and len(branches[0][1]) == len(S):
                    evaluations.append({"word": G[g], "cost": 1 + baseline_value, "exact": True,
                                        "skip_reason": "No information and no chance of winning"})
                    continue
                # Valid lower bounds allow skipping actions, never answer branches.
                items = [(mass, part, self.lower_bound(part)) for mass, part in branches]
                items.sort(key=lambda item: -item[0])
                bound = 1.0 + sum(mass * lb for mass, _, lb in items)
                pruned = bound >= best_cost - TIE_TOL
                if not pruned:
                    for mass, part, lb in items:
                        bound += mass * (self.value(part) - lb)
                        if bound >= best_cost - TIE_TOL:
                            pruned = True
                            break
                if pruned:
                    self.stats["pruned_actions"] += 1
                    evaluations.append({"word": G[g], "cost": None, "exact": False, "lower_bound": float(bound)})
                else:
                    best_g, best_cost = g, float(bound)
                    evaluations.append({"word": G[g], "cost": best_cost, "exact": True})
        return best_g, {
            "baseline_at_same_state": G[baseline], "selected": G[best_g],
            "changed": best_g != baseline, "candidates_remaining": len(S),
            "baseline_expected_remaining": baseline_value,
            "selected_action_then_baseline_expected_remaining": best_cost,
            "model_expected_reduction": float(baseline_value - best_cost),
            "model_dominance_verified": best_cost <= baseline_value + TIE_TOL,
            "actions": evaluations,
        }


def play(policy, answer, mode=None):
    """Reporting stops after 12 moves; the model's baseline value calculation does not."""
    S = np.flatnonzero(policy.w > 0)
    guesses, decisions = [], []
    g = INDEX["salet"]
    for turn in range(1, MAX_TURNS + 1):
        if turn > 1:
            use_rollout = mode == "all" or (mode == "second" and turn == 2)
            if use_rollout:
                g, detail = policy.rollout_action(S)
                decisions.append(dict(detail, turn=turn))
            else:
                g = policy.baseline_action(S)
        guesses.append(G[g])
        if g == answer:
            return {"turns": turn, "solved": True, "not_found": False, "capped": False,
                    "guesses": guesses, "rollout_decisions": decisions}
        S = S[Q[g, S] == Q[g, answer]]
        if not len(S):
            return {"turns": 99, "solved": False, "not_found": True, "capped": False,
                    "guesses": guesses, "rollout_decisions": decisions}
    return {"turns": MAX_TURNS + 1, "solved": False, "not_found": False, "capped": True,
            "guesses": guesses, "rollout_decisions": decisions}


def run_day(task):
    day, mode, top = task
    policy = DayPolicy(bt.weights("P2s", day), top=top)
    answer = bt.SOL[day]
    baseline = play(policy, answer)
    rollout = play(policy, answer, mode=mode)
    return day, {"baseline": baseline, "rollout": rollout, "cache_stats": dict(policy.stats)}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(per_day, days):
    summaries = {}
    for variant in ("baseline", "rollout"):
        rows = [per_day[d][variant] for d in days]
        solved = [r["turns"] for r in rows if r["solved"]]
        summaries[variant] = {
            "days": len(days), "solved": len(solved),
            "avg_solved": float(np.mean(solved)) if solved else None,
            "total_guesses_solved": sum(solved),
            "distribution_solved": {str(k): solved.count(k) for k in sorted(set(solved))},
            "failed_within_6": sum(r["turns"] > 6 for r in rows),
            "not_found": sum(r["not_found"] for r in rows),
            "capped_at_12": sum(r["capped"] for r in rows),
        }
    both = [d for d in days if per_day[d]["baseline"]["solved"] and per_day[d]["rollout"]["solved"]]
    diffs = np.array([per_day[d]["rollout"]["turns"] - per_day[d]["baseline"]["turns"] for d in both], float)
    mean = float(diffs.mean()) if len(diffs) else None
    halfwidth = float(1.96 * diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else None
    paired = {
        "both_solved_days": len(both), "mean_turn_difference": mean,
        "descriptive_normal_ci95": [mean - halfwidth, mean + halfwidth] if halfwidth is not None else None,
        "net_turns_saved": int(-diffs.sum()),
        "fewer_same_more": [int((diffs < 0).sum()), int((diffs == 0).sum()), int((diffs > 0).sum())],
    }
    decisions = [x for d in days for x in per_day[d]["rollout"]["rollout_decisions"]]
    model = {
        "rollout_decisions": len(decisions), "changed_decisions": sum(x["changed"] for x in decisions),
        "games_with_changes": sum(any(x["changed"] for x in per_day[d]["rollout"]["rollout_decisions"]) for d in days),
        "mean_local_model_reduction": float(np.mean([x["model_expected_reduction"] for x in decisions])) if decisions else 0.0,
        "note": "Local conditional action advantage assuming baseline continuation; not a measured overall score gain.",
    }
    return summaries, paired, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=96, help="Uniform sampled days; 0 means all available evaluation days.")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mode", choices=("second", "all"), default="second")
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--source-label", default="Local data/nyt_history.json; provenance to be verified")
    parser.add_argument("--output", help="Default results/pilot_rollout_<mode>.json")
    args = parser.parse_args()
    if args.limit < 0 or args.workers < 1 or args.top < 2:
        parser.error("Require limit >= 0, workers >= 1, and top >= 2")
    population = [d for d in bt.DAYS if "2025-01-01" <= d <= "2026-09-25"]
    if not population:
        raise RuntimeError("No evaluation dates in the required history window")
    limit = min(args.limit or len(population), len(population))
    days = sorted(random.Random(args.seed).sample(population, limit))
    history_path = os.path.join(DATA, "nyt_history.json")
    reference_path = os.path.join(RESULTS, "backtest_days.json")
    provenance = {
        "label": args.source_label, "history_path": history_path,
        "history_sha256": sha256_file(history_path),
        "prior_sha256": sha256_file(os.path.join(DATA, "tool_prior.json")),
        "reference_baseline_path": reference_path,
        "reference_baseline_sha256": sha256_file(reference_path) if os.path.exists(reference_path) else None,
    }
    t0 = time.time()
    print(f"Exact policy-rollout pilot: {len(days)}/{len(population)} days; mode={args.mode}; "
          f"top={args.top}; seed={args.seed}; workers={args.workers}; P2s/SALET; pure mean objective.", flush=True)
    per_day = {}
    with Pool(args.workers) as pool:
        for completed, (day, result) in enumerate(pool.imap_unordered(
            run_day, [(d, args.mode, args.top) for d in days], chunksize=1
        ), 1):
            per_day[day] = result
            if completed % 4 == 0 or completed == len(days):
                b = [r["baseline"]["turns"] for r in per_day.values() if r["baseline"]["solved"]]
                r = [r["rollout"]["turns"] for r in per_day.values() if r["rollout"]["solved"]]
                print(f"{completed}/{len(days)} days; partial baseline={np.mean(b):.4f}, rollout={np.mean(r):.4f}; "
                      f"{time.time()-t0:.1f}s", flush=True)
    summaries, paired, model = summarize(per_day, days)
    reference = {"available": os.path.exists(reference_path)}
    if reference["available"]:
        with open(reference_path) as f:
            saved = json.load(f)["P2s"]
        checked = [d for d in days if d in saved]
        mismatches = [{"day": d, "saved": saved[d], "rerun": per_day[d]["baseline"]["turns"]}
                      for d in checked if saved[d] != per_day[d]["baseline"]["turns"]]
        reference.update(checked_days=len(checked), mismatch_count=len(mismatches), mismatches=mismatches)
    out = {
        "exploratory": True, "caution": "Historical exploratory comparison; not an independent holdout claim.",
        "date_start": "2025-01-01", "date_end": "2026-09-25", "history_source": provenance,
        "population_days": len(population), "sample_days": len(days), "seed": args.seed, "days": days,
        "sampling": "Python random.Random(seed).sample without replacement, then chronological sorting",
        "mode": args.mode, "top": args.top, "prior": "P2s", "opener": "salet",
        "objective": "Expected total guesses under current day's fixed prior; no six-turn penalty",
        "model_evaluation": "All feedback branches and all positive-mass candidates; unchanged baseline continuation to completion",
        "reporting_turn_cap": MAX_TURNS, "summary": summaries, "paired": paired,
        "model_advantages": model, "reference_baseline": reference, "per_day": per_day,
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    path = args.output or os.path.join(RESULTS, "pilot_rollout_" + args.mode + ".json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print("summary", json.dumps(summaries), flush=True)
    print("paired", json.dumps(paired), flush=True)
    print("model_advantages", json.dumps(model), flush=True)
    print("reference_baseline", json.dumps(reference), flush=True)
    print(f"Wrote {os.path.abspath(path)} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
