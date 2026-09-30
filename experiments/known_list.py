"""Baseline: the textbook setting, where the solver KNOWS the 2,315-word answer list (uniform over it).

Usage:
  python3 experiments/known_list.py suggest [guess:feedback ...]   # g=green y=yellow b=gray, e.g. crane:gbbbg
  python3 experiments/known_list.py bench                          # compare strategies over all 2,315 answers
  env: TOPK (lookahead width, default 12), STARTS (openers for the lookahead, default crane,salet)
"""
import os
import sys
import time
from functools import lru_cache

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANSWERS = open(os.path.join(ROOT, "data", "answers.txt")).read().split()
GUESSES = open(os.path.join(ROOT, "data", "allowed.txt")).read().split()
G_INDEX = {w: i for i, w in enumerate(GUESSES)}
A_INDEX = {w: i for i, w in enumerate(ANSWERS)}
ANSWER_GUESS_IDX = np.array([G_INDEX[w] for w in ANSWERS])
IS_ANSWER = np.zeros(len(GUESSES), bool)
IS_ANSWER[ANSWER_GUESS_IDX] = True
ALL_GREEN = 242  # 2 + 2*3 + 2*9 + 2*27 + 2*81


def build_matrix():
    """P[g, a] = feedback code of guess g against answer a (base 3, 0 gray / 1 yellow / 2 green)."""
    path = os.path.join(ROOT, "cache", "patterns.npy")
    if os.path.exists(path):
        return np.load(path)
    g = np.array([[ord(c) - 97 for c in w] for w in GUESSES], np.int8)
    a = np.array([[ord(c) - 97 for c in w] for w in ANSWERS], np.int8)
    green = [g[:, None, i] == a[None, :, i] for i in range(5)]
    yellow = []
    for i in range(5):
        letter = g[:, i][:, None]
        avail = sum(((a[None, :, j] == letter) & ~green[j]).astype(np.int8) for j in range(5))
        used = sum(((g[:, k] == g[:, i])[:, None] & yellow[k]).astype(np.int8) for k in range(i))
        yellow.append(~green[i] & (avail - used > 0))
    code = sum((green[i] * 2 + yellow[i]).astype(np.uint8) * (3 ** i) for i in range(5))
    code = code.astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, code)
    return code


P = build_matrix()


def counts(S):
    """Pattern histogram of every guess against candidate set S (array of answer indices)."""
    sub = P[:, S].astype(np.int64) + (np.arange(len(GUESSES)) * 243)[:, None]
    return np.bincount(sub.ravel(), minlength=len(GUESSES) * 243).reshape(len(GUESSES), 243)


def entropy_scores(S):
    c = counts(S).astype(np.float64)
    p = c / len(S)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(p > 0, p * np.log2(p), 0), axis=1)
    return h


def candidate_mask(S):
    m = np.zeros(len(GUESSES), bool)
    m[ANSWER_GUESS_IDX[S]] = True
    return m


# ---------------- strategies: each maps candidate set S -> guess index ----------------

def strat_naive(S):
    """Just guess the first word still possible."""
    return int(ANSWER_GUESS_IDX[S[0]])


def strat_entropy(S):
    """Greedy max entropy over all allowed guesses; ties go to possible answers."""
    if len(S) <= 2:
        return int(ANSWER_GUESS_IDX[S[0]])
    h = entropy_scores(S) + candidate_mask(S) * 1e-6
    return int(np.argmax(h))


def strat_entropy_plus(S):
    """Entropy, but reward guesses that might win outright (approx. expected-guesses cost)."""
    if len(S) <= 2:
        return int(ANSWER_GUESS_IDX[S[0]])
    h = entropy_scores(S)
    n = len(S)
    p_win = candidate_mask(S) / n
    # remaining uncertainty after the guess, mapped to expected extra guesses by a rough fit
    rem = np.log2(n) - h
    extra = 1 + 0.55 * rem + 0.1 * rem ** 2
    cost = 1 + (1 - p_win) * extra
    return int(np.argmin(cost))


def partition(S, g):
    codes = P[g, S]
    order = np.argsort(codes, kind="stable")
    codes_sorted = codes[order]
    cuts = np.flatnonzero(np.diff(codes_sorted)) + 1
    parts = np.split(S[order], cuts)
    keys = codes_sorted[np.concatenate(([0], cuts))] if len(S) else []
    return list(zip(keys.tolist(), parts))


def make_lookahead(top_k=12):
    """Exact expected-guess search over the top_k entropy guesses (plus top candidates), memoised."""

    @lru_cache(maxsize=None)
    def best(key):
        S = np.array(key)
        n = len(S)
        if n == 1:
            return 1.0, int(ANSWER_GUESS_IDX[S[0]])
        if n == 2:
            return 1.5, int(ANSWER_GUESS_IDX[S[0]])
        h = entropy_scores(S)
        cand = candidate_mask(S)
        pool = set(np.argsort(-h)[:top_k].tolist())
        hc = np.where(cand, h, -1)
        pool |= set(np.argsort(-hc)[:top_k].tolist())
        best_cost, best_g = float("inf"), None
        for g in sorted(pool, key=lambda x: -h[x]):
            parts = partition(S, g)
            if len(parts) == 1 and parts[0][0] != ALL_GREEN:
                continue  # guess gives no information
            cost = 1.0
            for code, part in parts:
                if code == ALL_GREEN:
                    continue
                cost += len(part) / n * best(tuple(part.tolist()))[0]
                if cost >= best_cost:
                    break
            if cost < best_cost:
                best_cost, best_g = cost, g
        return best_cost, best_g

    def strat(S):
        return best(tuple(np.asarray(S).tolist()))[1]

    strat.best = best
    return strat


# ---------------- evaluation ----------------

def evaluate(strategy, first=None):
    """Build the full decision tree and return (average guesses, worst case, distribution)."""
    dist = {}

    def walk(S, depth, forced=None):
        g = forced if forced is not None else strategy(S)
        for code, part in partition(S, g):
            if code == ALL_GREEN:
                dist[depth] = dist.get(depth, 0) + len(part)
            else:
                walk(part, depth + 1)

    walk(np.arange(len(ANSWERS)), 1, G_INDEX[first] if first else None)
    total = sum(d * c for d, c in dist.items())
    return total / len(ANSWERS), max(dist), dict(sorted(dist.items()))


def parse_feedback(s):
    s = s.lower()
    table = {"g": 2, "2": 2, "🟩": 2, "y": 1, "1": 1, "🟨": 1, "b": 0, "0": 0, "x": 0, "⬜": 0, "⬛": 0}
    vals = [table[c] for c in s if c in table]
    assert len(vals) == 5, f"bad feedback {s!r}"
    return sum(v * 3 ** i for i, v in enumerate(vals))


def suggest(history):
    S = np.arange(len(ANSWERS))
    for item in history:
        word, fb = item.split(":")
        S = S[P[G_INDEX[word], S] == parse_feedback(fb)]
    print(f"{len(S)} possible answers:", " ".join(ANSWERS[i] for i in S[:40]), "..." if len(S) > 40 else "")
    if len(S) == 0:
        return
    if len(S) == 1:
        print("The answer is", ANSWERS[S[0]])
        return
    look = make_lookahead(12)
    cost, g = look.best(tuple(S.tolist()))
    h = entropy_scores(S)
    print(f"Suggest: {GUESSES[g]} (expected {cost:.2f} more guesses, {h[g]:.2f} bits, "
          f"{'could be the answer' if IS_ANSWER[g] and g in set(ANSWER_GUESS_IDX[S]) else 'probe word'})")
    top = np.argsort(-h)[:5]
    print("Top 5 by information:", ", ".join(f"{GUESSES[i]}({h[i]:.2f})" for i in top))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if cmd == "suggest":
        suggest(sys.argv[2:])
    elif cmd == "bench":
        t = time.time()
        h0 = entropy_scores(np.arange(len(ANSWERS)))
        print("Highest-information openers:", ", ".join(f"{GUESSES[i]}({h0[i]:.3f})" for i in np.argsort(-h0)[:8]))
        runs = [
            ("naive filter (guess first possible word), CRANE", strat_naive, "crane"),
            ("greedy entropy", strat_entropy, None),
            ("greedy entropy, CRANE", strat_entropy, "crane"),
            ("entropy + win chance, CRANE", strat_entropy_plus, "crane"),
        ]
        for name, s, first in runs:
            avg, worst, dist = evaluate(s, first)
            print(f"{name:<48} avg {avg:.4f}  worst {worst}  dist {dist}  ({time.time()-t:.0f}s)", flush=True)
        look = make_lookahead(int(os.environ.get("TOPK", 12)))
        for first in os.environ.get("STARTS", "crane,salet").split(","):
            avg, worst, dist = evaluate(look, first)
            print(f"{'lookahead search, ' + first.upper():<48} avg {avg:.4f}  worst {worst}  dist {dist}  ({time.time()-t:.0f}s)",
                  flush=True)
