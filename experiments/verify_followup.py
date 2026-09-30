"""Audit the archived follow-up experiments without network or paid API calls.

Default: independently recompute saved statistics, sampling, source hashes and
recorded path outcomes. --solver-checks additionally checks the alternative
formulas by direct enumeration, rollout by per-answer replay, the GINGS path,
and small safety certificates. Importing the solver builds its regenerable
168 MB feedback cache if absent. --full-baseline explicitly reruns 633 games.

Examples (run from the repository root):
  python experiments/verify_followup.py
  python experiments/verify_followup.py --solver-checks
  python experiments/verify_followup.py --solver-checks --full-baseline --workers 2

The old pilot_rollout_validation.json omitted its input state/weights. Its
numeric output is preserved as an old log; this script supplies a new fully
specified independent check, not a claimed reproduction of that missing state.
Wall-clock-limited safety searches are likewise not bit-for-bit reproducible.
"""

import argparse
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

os.environ["LAST_DAY"] = "2026-09-25"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXPECTED_HISTORY_SHA256 = "e9b4f1592b313cf03e8784c602d117ab49024834d7ad880a318a96d0230a4187"


def read(name):
    return json.loads((RESULTS / name).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-11), (actual, expected)


def feedback(guess, answer):
    """Independent two-pass implementation; consumes greens before yellows."""
    colors = [2 if g == a else 0 for g, a in zip(guess, answer)]
    left = Counter(a for a, color in zip(answer, colors) if color != 2)
    for i, letter in enumerate(guess):
        if colors[i] == 0 and left[letter]:
            colors[i] = 1
            left[letter] -= 1
    return sum(color * 3 ** i for i, color in enumerate(colors))


def counts(values):
    return {str(k): v for k, v in sorted(Counter(values).items())}


def paired(values, baseline):
    delta = [a - b for a, b in zip(values, baseline)]
    return {"net_turns_saved": -sum(delta),
            "fewer_same_more": [sum(d < 0 for d in delta), sum(d == 0 for d in delta),
                                 sum(d > 0 for d in delta)]}


def audit_saved():
    history_path = ROOT / "data/nyt_history.json"
    history = json.loads(history_path.read_text())
    history_sha = sha(history_path)
    assert history_sha == EXPECTED_HISTORY_SHA256
    dates = sorted(history)
    assert len(dates) == (date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days + 1
    allowed = set((ROOT / "data/allowed.txt").read_text().split())
    assert all(r["solution"] in allowed for r in history.values())
    canonical = read("backtest_days.json")["P2s"]
    full = read("baseline_uploaded_history_validation.json")
    assert full["history_sha256"] == history_sha
    assert full["per_day"] == canonical
    assert full["mismatch_count"] == 0 and full["mismatches"] == []
    values = list(canonical.values())
    assert full["days"] == len(values) == 633
    assert full["total_guesses"] == sum(values) == 1961
    close(full["mean_guesses"], statistics.mean(values))
    out = {"history": {"sha256": history_sha, "days": len(dates), "first": dates[0], "last": dates[-1]},
           "baseline": {"days": len(values), "total_guesses": sum(values),
                        "mean_guesses": statistics.mean(values), "distribution": counts(values),
                        "one_guess_fraction": values.count(1) / len(values),
                        "two_guesses_fraction": values.count(2) / len(values),
                        "at_most_three_fraction": sum(v <= 3 for v in values) / len(values),
                        "saved_day_scores_match": True}, "pilots": {}}
    path_count = 0

    def check_path(day, row):
        nonlocal path_count
        guesses = row["guesses"]
        assert row["solved"] and len(guesses) == row["turns"]
        assert all(word in allowed for word in guesses)
        answer = history[day]["solution"]
        codes = [feedback(g, answer) for g in guesses]
        assert codes[-1] == 242 and 242 not in codes[:-1]
        path_count += 1

    for name in ("pilot_branch_cost_original_96.json", "pilot_rollout_second_original_96.json",
                 "pilot_temporal_original_64.json"):
        payload = read(name)
        days = payload["days"]
        source_hash = payload.get("history_sha256") or payload["history_source"]["history_sha256"]
        assert source_hash == history_sha
        start = "2026-01-01" if "temporal" in name else "2025-01-01"
        population = [d for d in dates if start <= d <= "2026-09-25"]
        assert len(population) == payload["population_days"]
        assert days == sorted(random.Random(payload["seed"]).sample(population, len(days)))
        variants = list(payload["summary"])
        by_variant = (payload["per_day"] if "branch_cost" in name else
                      {v: {d: payload["per_day"][d][v] for d in days} for v in variants})
        base = [by_variant["baseline"][d]["turns"] for d in days]
        assert base == [canonical[d] for d in days]
        summaries = {}
        for variant, rows in by_variant.items():
            assert set(rows) == set(days)
            for day in days:
                check_path(day, rows[day])
            scores = [rows[d]["turns"] for d in days]
            saved = payload["summary"][variant]
            assert saved["days"] == len(scores) == saved["solved"]
            temporal = "temporal" in name
            assert saved["total_guesses_with_cap_penalty" if temporal else "total_guesses_solved"] == sum(scores)
            close(saved["mean_guesses_with_cap_penalty" if temporal else "avg_solved"], statistics.mean(scores))
            assert saved["distribution" if temporal else "distribution_solved"] == counts(scores)
            assert saved["over6" if temporal else "failed_within_6"] == sum(s > 6 for s in scores)
            comparison = paired(scores, base)
            if temporal:
                assert saved["net_turns_saved_vs_baseline"] == comparison["net_turns_saved"]
                assert saved["fewer_same_more"] == comparison["fewer_same_more"]
                assert saved["requested_openers"] == dict(Counter(rows[d]["requested_opener"] for d in days))
                assert saved["actual_openers"] == dict(Counter(rows[d]["guesses"][0] for d in days))
            elif variant != "baseline":
                saved_pair = payload["paired"][variant] if "branch_cost" in name else payload["paired"]
                assert all(saved_pair[k] == v for k, v in comparison.items())
            summaries[variant] = {"total_guesses": sum(scores), "mean_guesses": statistics.mean(scores), **comparison}
        out["pilots"][name] = {"days": len(days), "baseline_matches": True, "summary": summaries}
        if "rollout" in name:
            decisions = [r for d in days for r in by_variant["rollout"][d]["rollout_decisions"]]
            for row in decisions:
                reduction = row["baseline_expected_remaining"] - row["selected_action_then_baseline_expected_remaining"]
                close(reduction, row["model_expected_reduction"])
                assert reduction >= -1e-12 and row["model_dominance_verified"]
            model = payload["model_advantages"]
            assert model["changed_decisions"] == sum(r["changed"] for r in decisions)
            close(model["mean_local_model_reduction"], statistics.mean(r["model_expected_reduction"] for r in decisions))
        if "temporal" in name:
            assert payload["fits_sha256"] == sha(RESULTS / "temporal_prior.json")

    prior = read("temporal_prior.json")
    for name, digest in prior["source_hashes"].items():
        assert sha(ROOT / "data" / name) == digest
    prior_scores = {v: [] for v in ("base", "structural", "full")}
    for day, row in sorted(prior["per_day"].items()):
        assert row["truth_word"] == history[day]["solution"]
        fit = prior["fits"][row["cutoff"]]
        assert fit["train_end"] < fit["cutoff"] <= day
        assert fit["history_sha256"] == history_sha
        for variant in prior_scores:
            prior_scores[variant].append(row[variant + "_log2p"])
    for variant, scores in prior_scores.items():
        saved = prior["summary"][variant]
        gains = [a - b for a, b in zip(scores, prior_scores["base"])]
        assert saved["days"] == len(scores) == 268
        close(saved["mean_log2p"], statistics.mean(scores))
        close(saved["mean_log2p_gain_vs_base"], statistics.mean(gains))
        halfwidth = 1.96 * statistics.stdev(gains) / math.sqrt(len(gains))
        for a, b in zip(saved["descriptive_normal_ci95_gain"],
                        (statistics.mean(gains) - halfwidth, statistics.mean(gains) + halfwidth)):
            close(a, b)
    out["temporal_prediction"] = {"days": 268, "source_hashes_match": True,
                                  "training_dates_precede_predictions": True,
                                  "summary": prior["summary"]}
    safety = read("baseline_all_words_safety.json")
    distribution = {int(k): v for k, v in safety["visited_solved_turn_distribution"].items()}
    assert sum(distribution.values()) == safety["solved_answers"]
    assert sum(n for turn, n in distribution.items() if turn > 6) == safety["visited_solved_answers_over_six"]
    assert max(distribution) == safety["maximum_visited_solved_turns"]
    longest = safety["longest_visited_solved_path"]
    assert [feedback(g, longest["answer"]) for g in longest["guesses"]] == longest["feedback_codes"]
    assert longest["turns"] == len(longest["guesses"])
    for name in ("six_turn_salet_certificate.json", "six_turn_salet_fast_certificate.json"):
        result = read(name)
        assert result["allowed_sha256"] == sha(ROOT / "data/allowed.txt")
        assert result["status"] == "UNKNOWN" and result["policy"] is None
        assert result["independently_verified"] is False
    out["recorded_paths_checked"] = path_count
    out["path_check_scope"] = "Legal guesses, guess count, and terminal outcome checked using independent feedback; this alone does not revalidate action selection."
    out["saved_safety_scope"] = "Partial stress-traversal arithmetic and GINGS feedback checked; neither archived UNKNOWN search is a complete six-turn certificate."
    return out


def solver_checks():
    import numpy as np
    import backtest as bt
    from common import ALL_GREEN, G, INDEX, N, Q, choose
    from pilot_branch_cost import choose_alternative
    from pilot_rollout import DayPolicy
    from six_turn_guard import SixTurnGuard, verify_certificate

    words = ["ferns", "terns", "turns", "burns", "barns", "warns", "horns", "corns"]
    weights = [17, 13, 11, 7, 5, 3, 2, 1]
    w = np.zeros(N)
    for word, weight in zip(words, weights):
        w[INDEX[word]] = weight
    S = np.flatnonzero(w > 0)
    p = w[S] / w[S].sum()
    hs = -sum(float(x) * math.log2(float(x)) for x in p)
    costs = {key: np.empty(N) for key in ("baseline", "branch", "conditional")}
    # No bincount or entropy-difference shortcut for bucket entropies here.
    for g in range(N):
        buckets = {}
        for answer, probability in zip(S, p):
            code = feedback(G[g], G[int(answer)])
            assert code == int(Q[g, answer])
            buckets.setdefault(code, []).append(float(probability))
        masses = {code: sum(probs) for code, probs in buckets.items()}
        hfeedback = -sum(mass * math.log2(mass) for mass in masses.values())
        rem = max(hs - hfeedback, 0)
        fail = 1 - masses.get(ALL_GREEN, 0)
        phi = lambda h: 1 + .55 * h + .1 * h * h
        costs["baseline"][g] = 1 + fail * phi(rem)
        costs["conditional"][g] = 1 + fail * phi(rem / fail if fail else 0)
        continuation = 0.0
        for code, probs in buckets.items():
            if code != ALL_GREEN:
                mass = masses[code]
                entropy = -sum((v / mass) * math.log2(v / mass) for v in probs)
                continuation += mass * phi(entropy)
        costs["branch"][g] = 1 + continuation
    chosen = {"baseline": int(choose(S, w))}
    for variant in ("branch", "conditional"):
        chosen[variant], original = choose_alternative(S, w, variant)
        assert original == chosen["baseline"]
    for variant, g in chosen.items():
        close(float(costs[variant][g]), float(costs[variant].min()))

    def replay(answer, initial=S, first=None, weights=w):
        part = initial.copy()
        path = []
        while len(path) < len(initial) + 1:
            g = first if not path and first is not None else int(choose(part, weights))
            path.append(G[g])
            if g == answer:
                return path
            code = feedback(G[g], G[int(answer)])
            child = part[Q[g, part] == code]
            assert 0 < len(child) < len(part)
            part = child
        raise AssertionError("No solution within candidate-elimination bound")

    policy = DayPolicy(w, top=6)
    baseline = policy.baseline_action(S)
    paths = {G[int(a)]: replay(int(a)) for a in S}
    expected = sum(float(prob) * len(paths[G[int(a)]]) for a, prob in zip(S, p))
    close(policy.value(S), expected)
    candidate_costs = {}
    for g in policy.candidate_actions(S, baseline):
        candidate_costs[G[g]] = sum(float(prob) * len(replay(int(a), first=g)) for a, prob in zip(S, p))
    selected, detail = policy.rollout_action(S)
    close(detail["selected_action_then_baseline_expected_remaining"], min(candidate_costs.values()))
    close(candidate_costs[G[selected]], min(candidate_costs.values()))

    stress = read("baseline_all_words_safety.json")
    longpath = stress["longest_visited_solved_path"]
    full_weights = bt.weights("P2s", stress["prior_day"])
    actual_path = replay(INDEX[longpath["answer"]], np.flatnonzero(full_weights > 0),
                         INDEX[stress["opener"]], full_weights)
    assert actual_path == longpath["guesses"]
    original = set((ROOT / "data/answers.txt").read_text().split())
    history = json.loads((ROOT / "data/nyt_history.json").read_text())
    assert longpath["answer"] not in original
    assert longpath["answer"] not in {v["solution"] for v in history.values()}

    guard = SixTurnGuard(weights=w, max_seconds=5)
    certificate = guard.certify(S, remaining=len(S))
    assert certificate.status == "TRUE" and verify_certificate(S, len(S), certificate.policy)
    assert guard.certify(S[:2], remaining=1).status == "FALSE"
    assert not verify_certificate(S, len(S), {"guess": int(S[0]), "branches": {}})
    assert SixTurnGuard(max_nodes=0, max_seconds=None).certify(S, 3).status == "UNKNOWN"
    return {"fully_specified_inputs": dict(zip(words, weights)),
            "feedback_pairs_checked": N * len(S),
            "formula_minimizers": {v: G[g] for v, g in chosen.items()},
            "formula_check": "All legal guesses directly enumerated; implementation-selected actions attain minimum costs.",
            "rollout": {"baseline_paths": paths, "baseline_expected_guesses": expected,
                        "candidate_costs": candidate_costs, "selected": G[selected],
                        "exact_recursive_value_matches_independent_replay": True},
            "gings": {"prior_day": stress["prior_day"], "guesses": actual_path,
                      "strict_candidate_reduction": True, "in_original_answer_list": False,
                      "in_recorded_history": False, "hypothetical_allowed_answer_only": True},
            "guard": {"valid_small_certificate_verified": True, "incomplete_certificate_rejected": True,
                      "one_turn_impossibility_verified": True, "zero_budget_is_unknown": True}}


def full_baseline(workers):
    from multiprocessing import Pool
    import backtest as bt
    expected = read("backtest_days.json")["P2s"]
    with Pool(workers) as pool:
        actual = {d: n for _, d, n in pool.map(bt.play, [("P2s", d) for d in sorted(expected)])}
    assert actual == expected
    return {"days": len(actual), "mismatches": [], "per_day": actual,
            "total_guesses": sum(actual.values()), "mean_guesses": statistics.mean(actual.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-checks", action="store_true")
    parser.add_argument("--full-baseline", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", default=str(RESULTS / "followup_validation.json"))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    started = time.monotonic()
    out = {"saved_results": audit_saved(), "paid_api_calls": 0,
           "solver_checks_requested": args.solver_checks, "full_baseline_requested": args.full_baseline,
           "limitations": [
               "Old pilot_rollout_validation.json omitted state and weights; its exact input is unrecoverable from that file.",
               "Old 60-second stress traversal and 30-second certificate search omitted complete invocation metadata.",
               "UNKNOWN certificate searches establish neither feasibility nor impossibility.",
               "Saved pilot statistics are historical exploratory results, not a new independent holdout."]}
    if args.solver_checks:
        print("Checking direct formulas, independent rollout replay, GINGS, and small certificates...", flush=True)
        out["solver_checks"] = solver_checks()
    if args.full_baseline:
        print("Explicitly rerunning all 633 baseline dates...", flush=True)
        out["full_baseline_rerun"] = full_baseline(args.workers)
    out["elapsed_seconds"] = time.monotonic() - started
    out["status"] = "PASS"
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "output": str(target),
                      "recorded_paths_checked": out["saved_results"]["recorded_paths_checked"],
                      "elapsed_seconds": out["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
