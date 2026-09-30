"""Jev plays Wordle directly: each turn it sees the rules plus every previous guess and its colours,
and scores the words still consistent with that feedback; the top-scored word is the next guess.
The answer list is NOT given. Code only enforces the colour constraints (like the on-screen keyboard).

Also: a constraint-checking probe, where Jev itself must decide whether a word fits the feedback.

Needs `pip install typesafe-sdk` and TYPESAFE_API_KEY. The 200-game run took ~80 s and cost about $0.11.
usage: python3 experiments/jev_player.py [games]   (default 200)
"""
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from typesafe_sdk import Noul, TypeSafeClient

from common import ALL_GREEN, G, IS_ANS, Q, choose, load_scores, save_json
from jev_score_words import WORDLE_EXPLAINED

RULES = WORDLE_EXPLAINED
COLOR = {0: "gray", 1: "yellow", 2: "green"}
POOL = ThreadPoolExecutor(12)
cache, lock = {}, threading.Lock()
tokens = [0]


def describe(guess, code):
    parts = []
    for i, ch in enumerate(G[guess].upper()):
        parts.append(f"{ch}={COLOR[code // 3 ** i % 3]}")
        # position-by-position colours, left to right
    return f"{G[guess].upper()}: " + ", ".join(parts)


def state_for(history):
    return {"game": RULES,
            "previous_guesses": [describe(g, c) for g, c in history] or ["(none yet: this is the first guess)"],
            "note": "Every word asked about below is still consistent with all the feedback so far."}


def score_batch(state, words):
    with TypeSafeClient() as c:
        r = c.system_one(state=state, questions={
            G[w]: Noul(instructions=f"'{G[w].upper()}' is today's hidden Wordle answer.") for w in words})
    with lock:
        tokens[0] += r.usage.input_tokens
    return {w: r.nouls[G[w]].noul for w in words}


def jev_scores(history, S):
    key = tuple(history)
    with lock:
        if key in cache:
            return cache[key]
    state = state_for(history)
    out = {}
    for res in POOL.map(lambda b: score_batch(state, b), [S[i:i + 200] for i in range(0, len(S), 200)]):
        out.update(res)
    with lock:
        cache[key] = out
    return out


def play_jev(answer, max_turns=10):
    S, history = np.arange(len(G)), []
    for turn in range(1, max_turns + 1):
        sc = jev_scores(history, S.tolist())
        g = max(S.tolist(), key=lambda w: (sc[w], -w))
        if g == answer:
            return turn, [G[x] for x, _ in history] + [G[g]]
        code = int(Q[g, answer])
        history.append((g, code))
        S = S[Q[g, S] == code]
    return max_turns + 1, [G[x] for x, _ in history]


def depths_for(weights):
    """Per-answer guess counts for the (deterministic) solver with a given prior."""
    out = {}

    def walk(S, depth):
        g = choose(S, weights)
        codes = Q[g, S]
        for code in np.unique(codes):
            part = S[codes == code]
            if code == ALL_GREEN:
                if IS_ANS[g]:
                    out[g] = depth
            elif IS_ANS[part].any():
                walk(part, depth + 1)

    walk(np.flatnonzero(weights > 0), 1)
    return out


def paired(a, b, name):
    d = np.array(a, float) - np.array(b, float)
    rng = np.random.default_rng(0)
    boots = [rng.choice(d, len(d)).mean() for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se if se > 0 else float("inf")
    print(f"  Jev player - {name}: mean diff {d.mean():+.3f} guesses  95% CI [{lo:+.3f}, {hi:+.3f}]  t={t:+.2f}  "
          f"(Jev fewer/same/more: {(d<0).sum()}/{(d==0).sum()}/{(d>0).sum()})")
    return {"vs": name, "diff": float(d.mean()), "ci95": [float(lo), float(hi)], "t": float(t),
            "fewer_same_more": [int((d < 0).sum()), int((d == 0).sum()), int((d > 0).sum())]}


def constraint_probe(n=300, seed=3):
    """Can Jev itself tell whether a word is consistent with one round of feedback?"""
    rng = random.Random(seed)
    ans_idx = np.flatnonzero(IS_ANS)
    items = []
    while len(items) < n:
        a, g = int(rng.choice(ans_idx)), rng.randrange(len(G))
        code = int(Q[g, a])
        cons = np.flatnonzero(Q[g] == code)
        want = len(items) % 2 == 0
        pool = cons if want else np.flatnonzero(Q[g] != code)
        w = int(rng.choice(pool))
        items.append((g, code, w, bool(Q[g, w] == code)))

    def ask(it):
        g, code, w, truth = it
        with TypeSafeClient() as c:
            r = c.system_one(state={"game": RULES, "previous_guesses": [describe(g, code)]},
                             questions={"q": Noul(instructions=f"'{G[w].upper()}' is still consistent with all the "
                                                               f"feedback above, so it could be the hidden answer.")})
        with lock:
            tokens[0] += r.usage.input_tokens
        return r.nouls["q"].noul, truth

    res = list(POOL.map(ask, items))
    p = np.array([x for x, _ in res]); y = np.array([t for _, t in res])
    acc = ((p > 0.5) == y).mean()
    r = np.empty(len(p)); r[np.argsort(p, kind="stable")] = np.arange(1, len(p) + 1)
    auc = (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
    print(f"Constraint probe ({n} items, half consistent): accuracy {acc*100:.1f}%  AUC {auc:.3f}", flush=True)
    return {"items": n, "accuracy": float(acc), "auc": float(auc)}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    t0 = time.time()
    probe = constraint_probe()
    rng = random.Random(2026)
    sample = rng.sample(sorted(np.flatnonzero(IS_ANS).tolist()), n)
    results = list(ThreadPoolExecutor(6).map(play_jev, sample))
    jev = [d for d, _ in results]
    print(f"Jev plays ({n} games, no answer list): avg {np.mean(jev):.3f}  over 6: {sum(d > 6 for d in jev)}  "
          f"first guess {results[0][1][0].upper()}  {time.time()-t0:.0f}s  input tokens {tokens[0]}  "
          f"cost ${tokens[0]*0.042/1e6:.3f}", flush=True)
    for (d, path), a in list(zip(results, sample))[:5]:
        print("   e.g.", G[a].upper(), "<-", " -> ".join(x.upper() for x in path), f"({d})")
    base = {"solver, uniform prior (no list)": np.ones(len(G)),
            "solver + static Jev prior (no list)": np.maximum(load_scores("jev_rules_wordle"), 1e-4),
            "solver, knows the list": IS_ANS.astype(float)}
    comparisons = []
    for name, w in base.items():
        dd = depths_for(w)
        other = [dd[a] for a in sample]
        print(f"{name}: same {n} games avg {np.mean(other):.3f}", flush=True)
        comparisons.append(dict(paired(jev, other, name), other_avg=float(np.mean(other))))
    save_json("jev_player.json", {"probe": probe, "games": n, "jev_avg": float(np.mean(jev)),
                                  "jev_over6": int(sum(d > 6 for d in jev)), "input_tokens": tokens[0],
                                  "per_game": {G[a]: {"guesses": d, "path": p} for (d, p), a in zip(results, sample)},
                                  "comparisons": comparisons})
