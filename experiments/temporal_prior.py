"""Past-only quarterly temporal tilts of P2, with fixed features and regularization.

Six structural features redistribute probability within each P2 category. Two
separate rhythm features adjust the three category masses in the `full` variant.
Every fit uses the 365 days strictly before its cutoff, no earlier than 2022-11-07.
All softmax likelihoods are exact, with no negative sampling or word exclusion.

usage: python3 experiments/temporal_prior.py --output results/temporal_prior.json
"""
import argparse
import bisect
import datetime as dt
import hashlib
import json
import os
import time
from functools import lru_cache

os.environ["LAST_DAY"] = "2026-09-25"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import minimize

import backtest as bt
from common import DATA, G, N, RESULTS

L2 = 2.0
EDITOR_START = "2022-11-07"
CLASS_NAMES = ("unused_original", "repeat", "outside_unused")
STRUCTURAL_NAMES = (
    "same_first_letter_past7", "same_last_letter_past7", "same_last2_past7",
    "mean_unique_letter_jaccard_past3", "repeat_fraction_times_recent_shift",
    "difficulty_times_recent_shift",
)
CATEGORY_NAMES = ("repeat_gap_if_at_least8_seen", "outside_unused_gap_if_at_least8_seen")
FEATURE_DEFINITIONS = {
    STRUCTURAL_NAMES[0]: "Fraction of answers in the previous 7 calendar days sharing the candidate's first letter",
    STRUCTURAL_NAMES[1]: "Fraction of answers in the previous 7 calendar days sharing the candidate's last letter",
    STRUCTURAL_NAMES[2]: "Fraction of answers in the previous 7 calendar days sharing the candidate's final two letters",
    STRUCTURAL_NAMES[3]: "Mean Jaccard overlap of unique-letter sets with answers in the previous 3 calendar days",
    STRUCTURAL_NAMES[4]: "Candidate (5-unique_letters)/4 times (past3 mean minus past90 mean of that fraction)",
    STRUCTURAL_NAMES[5]: "Candidate clip(-log10(base_prior)/4,0,1) times (past3 mean minus past90 mean of that difficulty)",
    CATEGORY_NAMES[0]: "Repeat-class only: min(calendar days since latest repeat event,30)/30 - 0.5; zero until 8 prior repeat events",
    CATEGORY_NAMES[1]: "Outside-unused-class only: corresponding gap feature; zero until 8 prior outside-unused events",
}

# Static word representations contain no dated answer labels.
_LETTERS = np.array([[ord(c) - 97 for c in w] for w in G], dtype=np.int16)
_PRESENCE = np.zeros((N, 26), dtype=bool)
for _j in range(5):
    _PRESENCE[np.arange(N), _LETTERS[:, _j]] = True
_UNIQUE = _PRESENCE.sum(axis=1)
_REPEAT_FRACTION = (5 - _UNIQUE) / 4.0
_DIFFICULTY = np.clip(-np.log10(bt.PRIOR) / 4.0, 0.0, 1.0)
_LAST2 = 26 * _LETTERS[:, 3] + _LETTERS[:, 4]
_H_DAYS = list(bt.DAYS)
_H_ORDINALS = [dt.date.fromisoformat(d).toordinal() for d in _H_DAYS]
_H_ANSWERS = np.array([bt.SOL[d] for d in _H_DAYS], dtype=np.int64)

# Query-time comparisons/bisects are strictly before the requested date. A later
# first-use date or later event never changes an earlier-date feature or mask.
_FIRST_USE = np.full(N, np.iinfo(np.int64).max, dtype=np.int64)
_EVENTS = [[], [], []]
for _ord, _answer in zip(_H_ORDINALS, _H_ANSWERS):
    _category = 1 if _FIRST_USE[_answer] < _ord else 0 if bt.ORIG[_answer] else 2
    _EVENTS[_category].append(_ord)
    _FIRST_USE[_answer] = min(_FIRST_USE[_answer], _ord)

_TRAIN_CACHE = {}
_FIT_CACHE = {}


def quarter_cutoff(day):
    date = dt.date.fromisoformat(day)
    return dt.date(date.year, 1 + 3 * ((date.month - 1) // 3), 1).isoformat()


def _past_answers(day, window):
    end = dt.date.fromisoformat(day).toordinal()
    lo = bisect.bisect_left(_H_ORDINALS, end - window)
    hi = bisect.bisect_left(_H_ORDINALS, end)
    return _H_ANSWERS[lo:hi]


def category_masks(day):
    used = _FIRST_USE < dt.date.fromisoformat(day).toordinal()
    return (bt.ORIG & ~used, used, ~bt.ORIG & ~used)


def features(day):
    """Return float64 (N,6) structural features using only dates strictly before day."""
    past7 = _past_answers(day, 7)
    past3 = _past_answers(day, 3)
    past90 = _past_answers(day, 90)
    X = np.zeros((N, 6), dtype=np.float64)
    if len(past7):
        first = np.bincount(_LETTERS[past7, 0], minlength=26) / len(past7)
        last = np.bincount(_LETTERS[past7, 4], minlength=26) / len(past7)
        last2 = np.bincount(_LAST2[past7], minlength=676) / len(past7)
        X[:, 0] = first[_LETTERS[:, 0]]
        X[:, 1] = last[_LETTERS[:, 4]]
        X[:, 2] = last2[_LAST2]
    if len(past3):
        for answer in past3:
            intersection = _PRESENCE[:, _PRESENCE[answer]].sum(axis=1)
            X[:, 3] += intersection / (_UNIQUE + _UNIQUE[answer] - intersection)
        X[:, 3] /= len(past3)
    if len(past3) and len(past90):
        X[:, 4] = _REPEAT_FRACTION * (_REPEAT_FRACTION[past3].mean() - _REPEAT_FRACTION[past90].mean())
        X[:, 5] = _DIFFICULTY * (_DIFFICULTY[past3].mean() - _DIFFICULTY[past90].mean())
    return X


def category_features(day):
    """Return float64 (3,2) category logits features; gates count only past events."""
    ordinal = dt.date.fromisoformat(day).toordinal()
    X = np.zeros((3, 2), dtype=np.float64)
    for category, column in ((1, 0), (2, 1)):
        events = _EVENTS[category]
        count = bisect.bisect_left(events, ordinal)
        if count >= 8:
            gap = ordinal - events[count - 1]
            X[category, column] = min(gap, 30) / 30.0 - 0.5
    return X


def _training_day(day):
    if day not in _TRAIN_CACHE:
        base = bt.weights("P2s", day)
        masks = category_masks(day)
        answer = bt.SOL[day]
        category = next(c for c, mask in enumerate(masks) if mask[answer])
        shares = np.array([base[mask].sum() for mask in masks])
        if np.any(shares <= 0) or np.any(base <= 0):
            raise ValueError("The selected training period must have positive P2 mass in all categories")
        X = features(day)
        conditional = base[masks[category]] / shares[category]
        _TRAIN_CACHE[day] = {
            "X": np.ascontiguousarray(X[masks[category]]), "log_base": np.log(conditional),
            "truth_x": X[answer].copy(), "truth_log_base": float(np.log(base[answer] / shares[category])),
            "category": category, "log_shares": np.log(shares / shares.sum()),
            "category_x": category_features(day),
        }
    return _TRAIN_CACHE[day]


def _optimize(objective, start):
    result = minimize(objective, np.asarray(start, dtype=np.float64), jac=True, method="L-BFGS-B",
                      options={"maxiter": 200, "ftol": 1e-11, "gtol": 1e-7})
    zero_loss, _ = objective(np.zeros(len(start)))
    penalty = 0.5 * L2 * float(result.x @ result.x)
    diagnostics = {
        "success": bool(result.success), "message": str(result.message), "iterations": int(result.nit),
        "function_evaluations": int(result.nfev), "gradient_max_abs": float(np.max(np.abs(result.jac))),
        "base_training_loglik_nats": float(-zero_loss),
        "fitted_training_loglik_nats": float(-result.fun + penalty),
        "penalized_training_loglik_nats": float(-result.fun), "l2_penalty": penalty,
    }
    if not result.success:
        raise RuntimeError("Temporal-prior optimization did not converge: " + json.dumps(diagnostics))
    return result.x.tolist(), diagnostics


@lru_cache(maxsize=1)
def source_hashes():
    out = {}
    for name in ("nyt_history.json", "tool_prior.json", "allowed.txt", "answers.txt"):
        path = os.path.join(DATA, name)
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        out[name] = digest.hexdigest()
    return out


def fit(cutoff, initial=None):
    """Fit only [max(cutoff-365d, editor start), cutoff); return serializable coefficients."""
    if cutoff in _FIT_CACHE:
        return _FIT_CACHE[cutoff]
    cutoff_date = dt.date.fromisoformat(cutoff)
    start = max(EDITOR_START, (cutoff_date - dt.timedelta(days=365)).isoformat())
    days = [d for d in _H_DAYS if start <= d < cutoff]
    if not days:
        raise ValueError("No eligible training days before cutoff")
    if initial is not None and initial["cutoff"] > cutoff:
        raise ValueError("A warm start must not come from a later cutoff")
    rows = [_training_day(d) for d in days]
    lengths = np.array([len(row["log_base"]) for row in rows], dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    X = np.concatenate([row["X"] for row in rows], axis=0)
    log_base = np.concatenate([row["log_base"] for row in rows])
    truth_x_sum = np.sum([row["truth_x"] for row in rows], axis=0)
    truth_log_base_sum = sum(row["truth_log_base"] for row in rows)

    def structural_objective(beta):
        logits = log_base + X @ beta
        maxima = np.maximum.reduceat(logits, starts)
        exponentials = np.exp(logits - np.repeat(maxima, lengths))
        totals = np.add.reduceat(exponentials, starts)
        probabilities = exponentials / np.repeat(totals, lengths)
        nll = np.sum(maxima + np.log(totals)) - truth_log_base_sum - truth_x_sum @ beta
        gradient = X.T @ probabilities - truth_x_sum + L2 * beta
        return float(nll + 0.5 * L2 * (beta @ beta)), gradient

    structural_start = initial["structural_beta"] if initial is not None else np.zeros(6)
    structural_beta, structural_diag = _optimize(structural_objective, structural_start)
    structural_diag["candidate_rows"] = int(len(log_base))

    group_X = np.array([row["category_x"] for row in rows])
    group_log_base = np.array([row["log_shares"] for row in rows])
    targets = np.array([row["category"] for row in rows])
    day_index = np.arange(len(rows))

    def category_objective(beta):
        logits = group_log_base + group_X @ beta
        maxima = logits.max(axis=1)
        exponentials = np.exp(logits - maxima[:, None])
        totals = exponentials.sum(axis=1)
        probabilities = exponentials / totals[:, None]
        nll = np.sum(maxima + np.log(totals) - logits[day_index, targets])
        residual = probabilities.copy()
        residual[day_index, targets] -= 1
        gradient = np.einsum("dc,dcf->f", residual, group_X) + L2 * beta
        return float(nll + 0.5 * L2 * (beta @ beta)), gradient

    category_start = initial["category_beta"] if initial is not None else np.zeros(2)
    category_beta, category_diag = _optimize(category_objective, category_start)
    result = {
        "cutoff": cutoff, "train_start": days[0], "train_end": days[-1], "train_days": len(days),
        "l2": L2, "structural_beta": structural_beta, "category_beta": category_beta,
        "structural_diagnostics": structural_diag, "category_diagnostics": category_diag,
        "history_sha256": source_hashes()["nyt_history.json"],
        "structural_features": list(STRUCTURAL_NAMES), "category_features": list(CATEGORY_NAMES),
    }
    _FIT_CACHE[cutoff] = result
    return result


def _softmax(logits):
    values = np.exp(logits - np.max(logits))
    return values / values.sum()


def weights(day, variant="full", fit_result=None):
    """All N positive weights; explicit fit_result is a serialized fit(cutoff) result."""
    if variant not in ("base", "structural", "full"):
        raise ValueError(variant)
    base = bt.weights("P2s", day)
    base = base / base.sum()
    if variant == "base":
        return base
    model = fit_result if fit_result is not None else fit(quarter_cutoff(day))
    if model["cutoff"] > day or model["train_end"] >= model["cutoff"]:
        raise ValueError("Fit must use only days before its cutoff, and cutoff must be no later than prediction day")
    X = features(day)
    structural_logits = X @ np.asarray(model["structural_beta"], dtype=np.float64)
    masks = category_masks(day)
    shares = np.array([base[mask].sum() for mask in masks])
    if np.any(shares <= 0):
        raise ValueError("Expected all three P2 categories to have positive mass")
    if variant == "full":
        shares = _softmax(np.log(shares) + category_features(day) @ np.asarray(model["category_beta"], dtype=np.float64))
    out = np.zeros(N, dtype=np.float64)
    for share, mask in zip(shares, masks):
        out[mask] = share * _softmax(np.log(base[mask]) + structural_logits[mask])
    if not np.all(np.isfinite(out)) or not np.all(out > 0) or not np.isclose(out.sum(), 1.0, atol=1e-12):
        raise AssertionError("Temporal prior must retain finite positive mass for every allowed word")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-09-25")
    parser.add_argument("--output", default=os.path.join(RESULTS, "temporal_prior.json"))
    args = parser.parse_args()
    days = [d for d in _H_DAYS if args.start <= d <= args.end]
    if not days:
        parser.error("No known evaluation dates in the requested range")
    t0 = time.time()
    fits = {}
    previous = None
    for cutoff in sorted({quarter_cutoff(d) for d in days}):
        print(f"Fitting cutoff {cutoff}, preceding 365 days, L2={L2}", flush=True)
        model = fit(cutoff, initial=previous)
        fits[cutoff] = model
        previous = model
        print(f"  structural={model['structural_beta']}; category={model['category_beta']}; "
              f"{time.time()-t0:.1f}s", flush=True)
    per_day = {}
    for i, day in enumerate(days, 1):
        cutoff = quarter_cutoff(day)
        answer = bt.SOL[day]
        category = next(c for c, mask in enumerate(category_masks(day)) if mask[answer])
        row = {"cutoff": cutoff, "truth_word": G[answer], "category": CLASS_NAMES[category]}
        for variant in ("base", "structural", "full"):
            w = weights(day, variant=variant, fit_result=fits[cutoff])
            row[variant + "_log2p"] = float(np.log2(w[answer]))
        per_day[day] = row
        if i % 32 == 0 or i == len(days):
            print(f"Scored {i}/{len(days)} days; {time.time()-t0:.1f}s", flush=True)
    summary = {}
    for variant in ("base", "structural", "full"):
        scores = np.array([per_day[d][variant + "_log2p"] for d in days])
        base = np.array([per_day[d]["base_log2p"] for d in days])
        diff = scores - base
        halfwidth = float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else None
        summary[variant] = {
            "days": len(days), "mean_log2p": float(scores.mean()), "mean_log2p_gain_vs_base": float(diff.mean()),
            "descriptive_normal_ci95_gain": [float(diff.mean()) - halfwidth, float(diff.mean()) + halfwidth]
            if halfwidth is not None else None,
        }
    result = {
        "exploratory": True, "evaluation_start": days[0], "evaluation_end": days[-1],
        "source_hashes": source_hashes(), "l2": L2,
        "regularization_definition": "sum historical log likelihood minus 0.5 * 2.0 * squared coefficient norm",
        "training_window_days": 365, "earliest_training_date": EDITOR_START,
        "refit_schedule": "calendar quarter start, using only earlier days",
        "feature_definitions": FEATURE_DEFINITIONS,
        "structural_features": list(STRUCTURAL_NAMES), "category_features": list(CATEGORY_NAMES),
        "variants": {"structural": "six features within each category, preserving P2 category masses",
                     "full": "structural tilt plus separately fitted two-feature category softmax"},
        "caution": "Past-only fitting within an exploratory historical evaluation; log-likelihood gains do not imply fewer guesses.",
        "fits": fits, "per_day": per_day, "summary": summary,
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print("summary", json.dumps(summary), flush=True)
    print(f"Wrote {os.path.abspath(args.output)} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
