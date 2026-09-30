"""Fixed-sample comparison of temporal priors and heuristic adaptive openers.

Optional --certificate requires a verified six-turn tree for all allowed words.
Every departure from that tree is separately certified, or the tree is followed.
No true answer is passed to the action selector or safety prover.
"""
import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter
from multiprocessing import Pool

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import backtest as bt
import temporal_prior as tp
from common import ALL_GREEN, DATA, G, INDEX, N, Q, RESULTS, choose
from six_turn_guard import SixTurnGuard, verify_certificate

VARIANTS = ("baseline", "structural_salet", "full_salet", "base_adaptive", "full_adaptive")
FITS = None
ROOT_POLICY = None
GUARD = None


def initialize(fits, root_policy):
    global FITS, ROOT_POLICY, GUARD
    FITS, ROOT_POLICY = fits, root_policy
    GUARD = SixTurnGuard(max_seconds=0.5, max_guesses=100_000) if root_policy else None


def play(w, answer, opener, guarded):
    S = np.arange(N, dtype=np.int64)
    proof = ROOT_POLICY if guarded else None
    guesses, checks = [], []
    for turn in range(1, 13):
        proposed = opener if turn == 1 else int(choose(S, w))
        if guarded:
            if int(proof["guess"]) != proposed:
                GUARD.weights = w
                result = GUARD.certify_action(S, 7 - turn, proposed)
                checks.append({"turn": turn, "proposed": G[proposed], "status": result.status})
                if result.status == "TRUE":
                    assert verify_certificate(S, 7 - turn, result.policy)
                    proof = result.policy
            g = int(proof["guess"])
        else:
            g = proposed
            row = Q[g, S]
            if np.all(row == row[0]) and row[0] != ALL_GREEN:
                g = int(S[np.argmax(w[S])])
        guesses.append(G[g])
        if g == answer:
            return {"turns": turn, "solved": True, "guesses": guesses,
                    "requested_opener": G[opener], "safety_checks": checks}
        code = int(Q[g, answer])
        S = S[Q[g, S] == code]
        assert len(S), "Positive-support answer was incorrectly excluded"
        if guarded:
            assert turn < 6, "Verified certificate failed six-turn guarantee"
            proof = proof["branches"][str(code)]
    return {"turns": 13, "solved": False, "guesses": guesses,
            "requested_opener": G[opener], "safety_checks": checks}


def run_day(day):
    fit = FITS[tp.quarter_cutoff(day)]
    assert fit["train_end"] < day
    ws = {"base": bt.weights("P2s", day),
          "structural": tp.weights(day, "structural", fit_result=fit),
          "full": tp.weights(day, "full", fit_result=fit)}
    assert all(np.isfinite(w).all() and np.all(w > 0) for w in ws.values())
    openers = {"base": int(choose(np.arange(N), ws["base"])),
               "full": int(choose(np.arange(N), ws["full"]))}
    definitions = {
        "baseline": ("base", INDEX["salet"], False),
        "structural_salet": ("structural", INDEX["salet"], ROOT_POLICY is not None),
        "full_salet": ("full", INDEX["salet"], ROOT_POLICY is not None),
        "base_adaptive": ("base", openers["base"], ROOT_POLICY is not None),
        "full_adaptive": ("full", openers["full"], ROOT_POLICY is not None),
    }
    if ROOT_POLICY:
        definitions["guarded_baseline"] = ("base", INDEX["salet"], True)
    rows = {v: play(ws[k], bt.SOL[day], opener, guarded)
            for v, (k, opener, guarded) in definitions.items()}
    return day, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", default=os.path.join(RESULTS, "temporal_prior.json"))
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--certificate", help="JSON containing a verified root 'policy' for every allowed word")
    parser.add_argument("--output", default=os.path.join(RESULTS, "pilot_temporal_64.json"))
    args = parser.parse_args()
    if args.limit <= 0 or args.workers <= 0:
        parser.error("limit and workers must be positive")
    with open(args.fits) as f:
        payload = json.load(f)
    history_hash = hashlib.sha256(open(os.path.join(DATA, "nyt_history.json"), "rb").read()).hexdigest()
    assert payload["source_hashes"]["nyt_history.json"] == history_hash
    root_policy = None
    if args.certificate:
        with open(args.certificate) as f:
            root_policy = json.load(f)["policy"]
        assert verify_certificate(np.arange(N), 6, root_policy)
        assert root_policy["guess"] == INDEX["salet"]
    population = [d for d in bt.DAYS if "2026-01-01" <= d <= "2026-09-25"]
    days = sorted(random.Random(args.seed).sample(population, min(args.limit, len(population))))
    started = time.time()
    per_day = {}
    print(f"Temporal pilot: {len(days)} days; seed={args.seed}; certified={root_policy is not None}", flush=True)
    with Pool(args.workers, initializer=initialize, initargs=(payload["fits"], root_policy)) as pool:
        for day, rows in pool.imap_unordered(run_day, days, chunksize=1):
            per_day[day] = rows
            if len(per_day) % 8 == 0 or len(per_day) == len(days):
                print(f"{len(per_day)}/{len(days)} days; {time.time()-started:.1f}s", flush=True)
    variants = list(per_day[days[0]])
    summary = {}
    for v in variants:
        rows = [per_day[d][v] for d in days]
        totals = np.array([r["turns"] for r in rows])
        diff = totals - np.array([per_day[d]["baseline"]["turns"] for d in days])
        summary[v] = {
            "days": len(days), "solved": sum(r["solved"] for r in rows),
            "mean_guesses_with_cap_penalty": float(totals.mean()),
            "total_guesses_with_cap_penalty": int(totals.sum()),
            "over6": int(np.sum(totals > 6)), "distribution": dict(Counter(map(int, totals))),
            "net_turns_saved_vs_baseline": int(-diff.sum()),
            "fewer_same_more": [int(np.sum(diff < 0)), int(np.sum(diff == 0)), int(np.sum(diff > 0))],
            "requested_openers": dict(Counter(r["requested_opener"] for r in rows)),
            "actual_openers": dict(Counter(r["guesses"][0] for r in rows)),
            "guard_checks": dict(Counter(c["status"] for r in rows for c in r["safety_checks"])),
        }
    with open(os.path.join(RESULTS, "backtest_days.json")) as f:
        saved = json.load(f)["P2s"]
    mismatches = [d for d in days if saved[d] != per_day[d]["baseline"]["turns"]]
    out = {
        "exploratory": True, "independent_holdout": False,
        "days": days, "seed": args.seed, "population_days": len(population),
        "fits_file": os.path.abspath(args.fits),
        "history_sha256": hashlib.sha256(open(os.path.join(DATA, "nyt_history.json"), "rb").read()).hexdigest(),
        "fits_sha256": hashlib.sha256(open(args.fits, "rb").read()).hexdigest(),
        "adaptive_opening_rule": "Original common.choose over all allowed words under the day's specified prior; heuristic, not full-tree optimal",
        "all_candidates_have_positive_probability": True,
        "certified_six_turn_policy_for_new_variants": root_policy is not None,
        "certificate_path": os.path.abspath(args.certificate) if args.certificate else None,
        "guard_rule": "Only adopt a proposed action with an exact complete six-turn continuation certificate; else follow the existing certificate" if root_policy else None,
        "baseline_mismatches": mismatches, "summary": summary, "per_day": per_day,
        "elapsed_seconds": time.time() - started,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(json.dumps(summary, indent=2), flush=True)
    print("baseline mismatches", mismatches, flush=True)
    print("Wrote", os.path.abspath(args.output), flush=True)


if __name__ == "__main__":
    main()
