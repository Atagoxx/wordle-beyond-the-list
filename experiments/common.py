"""Shared pieces: word lists, the feedback matrix, the one-step guess rule and the full-tree evaluator.

Setting used everywhere except known_list.py: the solver does NOT know the 2,315-word answer list.
Every one of the 12,972 accepted guesses is a candidate answer, weighted by a prior.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PRIORS = os.path.join(DATA, "priors")
RESULTS = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "cache")

G = open(os.path.join(DATA, "allowed.txt")).read().split()      # all accepted guesses, sorted
ANS = set(open(os.path.join(DATA, "answers.txt")).read().split())  # the original 2,315 answers
N = len(G)
INDEX = {w: i for i, w in enumerate(G)}
IS_ANS = np.array([w in ANS for w in G])
ALL_GREEN = 242  # feedback code = sum(color_i * 3**i), gray 0 / yellow 1 / green 2


def build_full():
    """Q[g, a] = feedback code of guess g against answer a, for all 12,972 x 12,972 pairs (168 MB, cached)."""
    path = os.path.join(CACHE, "patterns_full.npy")
    if os.path.exists(path):
        return np.load(path)
    g = np.array([[ord(c) - 97 for c in w] for w in G], np.int8)
    out = np.empty((N, N), np.uint8)
    for s in range(0, N, 1000):
        a = g[s:s + 1000]
        green = [g[:, None, i] == a[None, :, i] for i in range(5)]
        yellow = []
        for i in range(5):
            letter = g[:, i][:, None]
            avail = sum(((a[None, :, j] == letter) & ~green[j]).astype(np.int8) for j in range(5))
            used = sum(((g[:, k] == g[:, i])[:, None] & yellow[k]).astype(np.int8) for k in range(i))
            yellow.append(~green[i] & (avail - used > 0))
        out[:, s:s + 1000] = sum((green[i] * 2 + yellow[i]).astype(np.uint8) * (3 ** i) for i in range(5))
    os.makedirs(CACHE, exist_ok=True)
    np.save(path, out)
    return out


Q = build_full()


# ---------------- priors ----------------

def load_scores(name):
    """Per-word scores from data/priors/<name>.json, in G order."""
    d = json.load(open(os.path.join(PRIORS, name + ".json")))
    return np.array([d[w] for w in G], float)


def rank(s):
    return np.argsort(np.argsort(np.asarray(s, float), kind="stable"), kind="stable")


# Every prior is mapped onto the same value curve (the sorted probabilities of the first Jev prior),
# so priors differ only in how they ORDER the words. This keeps comparisons between priors fair.
CURVE = np.sort(load_scores("jev_rules_wordle"))


def matched(scores, power=1.0):
    return np.maximum(CURVE[rank(scores)], 1e-4) ** power


def tool_prior():
    """The base prior shipped in the web tool (packed as 2 base-36 chars per word: v = round(-100*log10(w)))."""
    d = json.load(open(os.path.join(DATA, "tool_prior.json")))
    v = np.array([int(d["weights"][i:i + 2], 36) for i in range(0, len(d["weights"]), 2)])
    return 10 ** (-v / 100)


# ---------------- guess rule ----------------

def choose(S, w, c1=0.55, c2=0.1):
    """One-step rule: minimise an estimated expected number of guesses.

    cost(g) = 1 + (1 - P(g is the answer)) * (1 + c1*rem + c2*rem^2),   c1 = 0.55, c2 = 0.1 (set by hand),
    rem = entropy of the candidates (bits) - expected information from g (bits).
    """
    p = w[S] / w[S].sum()
    if len(S) <= 2:
        return S[np.argmax(p)]
    hg = np.empty(N)
    for s in range(0, N, 1500):
        sub = Q[s:s + 1500][:, S].astype(np.int64) + (np.arange(min(1500, N - s)) * 243)[:, None]
        m = np.bincount(sub.ravel(), weights=np.broadcast_to(p, sub.shape).ravel(),
                        minlength=sub.shape[0] * 243).reshape(-1, 243)
        with np.errstate(divide="ignore", invalid="ignore"):
            hg[s:s + 1500] = -np.where(m > 0, m * np.log2(m), 0).sum(1)
    hs = -(p * np.log2(p, where=p > 0, out=np.zeros_like(p))).sum()
    pwin = np.zeros(N)
    pwin[S] = p
    rem = np.maximum(hs - hg, 0)
    cost = 1 + (1 - pwin) * (1 + c1 * rem + c2 * rem ** 2)
    return int(np.argmin(cost))


def evaluate(w, restrict_to_answers=False, first=None):
    """Play every one of the 2,315 original answers with prior w (deterministic, so no sampling error).

    Returns (average guesses, games over 6, distribution, first guess).
    """
    dist = {}
    S0 = np.flatnonzero(IS_ANS) if restrict_to_answers else np.flatnonzero(w > 0)
    first_seen = [None]

    def walk(S, depth, forced=None):
        g = forced if forced is not None else choose(S, w)
        if first_seen[0] is None:
            first_seen[0] = G[g]
        codes = Q[g, S]
        for code in np.unique(codes):
            part = S[codes == code]
            if code == ALL_GREEN:
                if IS_ANS[g]:
                    dist[depth] = dist.get(depth, 0) + 1
                continue
            if IS_ANS[part].any():
                walk(part, depth + 1)

    walk(S0, 1, INDEX[first] if first else None)
    solved = sum(dist.values())
    assert solved == IS_ANS.sum(), (solved, IS_ANS.sum())
    avg = sum(d * c for d, c in dist.items()) / solved
    fails = sum(c for d, c in dist.items() if d > 6)
    return avg, fails, dict(sorted(dist.items())), first_seen[0]


def save_json(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=1)
