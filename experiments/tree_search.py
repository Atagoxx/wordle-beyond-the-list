"""Near-exhaustive decision-tree search under the prior (no answer list).

Objective at a node S reached at depth d (d = index of the guess about to be made):
    f(S, d) = expected guesses from here + LAMBDA * P(finishing after guess 6)
with probabilities from the prior restricted to S. Every guess tried is followed through all of its
feedback branches exactly; only the set of guesses tried per node is limited (top-K by heuristic).

Prior: the tool's base prior (no history). ~10-14 min per opener on 4 cores.
usage: python3 experiments/tree_search.py <opener>        env: LAMBDA (default 5), PROCS (default 4)
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from common import ALL_GREEN, G, IS_ANS, N, Q, RESULTS, save_json, tool_prior

W = tool_prior()
LAMBDA = float(os.environ.get("LAMBDA", 5))
ROWS = np.arange(N)
memo = {}


def heuristic(S, p, k):
    """Top-k guesses by the one-step cost used in the tool, plus the most likely candidates."""
    hg = np.empty(N)
    for s in range(0, N, 2000):
        sub = Q[s:s + 2000][:, S].astype(np.int64) + (np.arange(min(2000, N - s)) * 243)[:, None]
        m = np.bincount(sub.ravel(), weights=np.broadcast_to(p, sub.shape).ravel(),
                        minlength=sub.shape[0] * 243).reshape(-1, 243)
        with np.errstate(divide="ignore", invalid="ignore"):
            hg[s:s + 2000] = -np.where(m > 0, m * np.log2(m), 0).sum(1)
    hs = -(p * np.log2(p)).sum()
    pwin = np.zeros(N)
    pwin[S] = p
    rem = np.maximum(hs - hg, 0)
    cost = 1 + (1 - pwin) * (1 + 0.55 * rem + 0.1 * rem ** 2)
    order = list(np.lexsort((ROWS, cost))[:k])
    for c in S[np.argsort(-p, kind="stable")[:2]]:
        if c not in order:
            order.append(int(c))
    return order


def lower_bound(n, pmax, d):
    if n == 1:
        return 1 + (LAMBDA if d > 6 else 0)
    return 1 + (1 - pmax) + (LAMBDA * (1 - pmax) if d + 1 > 6 else 0)


def solve(S, d):
    key = (S.tobytes(), d)
    hit = memo.get(key)
    if hit is not None:
        return hit
    n = len(S)
    wS = W[S]
    p = wS / wS.sum()
    if n == 1:
        res = (1 + (LAMBDA if d > 6 else 0), int(S[0]))
    elif n == 2:
        i = int(np.argmax(p))
        res = (1 + p[i] * (LAMBDA if d > 6 else 0) + p[1 - i] * (1 + (LAMBDA if d + 1 > 6 else 0)), int(S[i]))
    else:
        k = 4 if n > 1500 else 6 if n > 300 else 10 if n > 40 else 16
        best = (float("inf"), None)
        for g in heuristic(S, p, k):
            codes = Q[g, S]
            order = np.argsort(codes, kind="stable")
            cs = codes[order]
            cuts = np.flatnonzero(np.diff(cs)) + 1
            parts = np.split(S[order], cuts)
            pparts = np.split(p[order], cuts)
            keys = cs[np.concatenate(([0], cuts))]
            if len(parts) == 1 and keys[0] != ALL_GREEN:
                continue
            base = 1.0
            items = []
            for code, part, pp in zip(keys, parts, pparts):
                if code == ALL_GREEN:
                    base += pp.sum() * (LAMBDA if d > 6 else 0)
                else:
                    items.append((pp.sum(), part, pp.max()))
            items.sort(key=lambda x: -x[0])
            lb = [m * lower_bound(len(part), pm / m, d + 1) for m, part, pm in items]
            total = base + sum(lb)
            if total >= best[0]:
                continue
            for j, (m, part, pm) in enumerate(items):
                total += m * solve(part, d + 1)[0] - lb[j]
                if total >= best[0]:
                    break
            if total < best[0]:
                best = (total, int(g))
        res = best
    memo[key] = res
    return res


def solve_bucket(args):
    S, d = args
    t = time.time()
    c, g = solve(S, d)

    def walk(S, d):
        c, g = solve(S, d)
        node = {"g": G[g]}
        if len(S) > 1:
            codes = Q[g, S]
            for code in np.unique(codes):
                if code != ALL_GREEN:
                    node[str(int(code))] = walk(S[codes == code], d + 1)
        return node

    return c, walk(S, d), time.time() - t


def evaluate_tree(first, tree):
    """Guesses needed for each true answer when following the tree."""
    depths = []
    for a in np.flatnonzero(IS_ANS):
        node, g, d = tree, G.index(first), 1
        while g != a:
            code = str(int(Q[g, a]))
            node = node[code]
            g, d = G.index(node["g"]), d + 1
        depths.append(d)
    return np.array(depths)


if __name__ == "__main__":
    first = sys.argv[1]
    procs = int(os.environ.get("PROCS", 4))
    t0 = time.time()
    g0 = G.index(first)
    S0 = np.arange(N)
    codes = Q[g0, S0]
    buckets = [(S0[codes == c], 2) for c in np.unique(codes) if c != ALL_GREEN]
    buckets.sort(key=lambda b: -len(b[0]))
    p0 = W / W.sum()
    with Pool(procs) as pool:
        results = pool.map(solve_bucket, buckets, chunksize=1)
    tree = {"g": first}
    total = 1.0
    for (S, _), (c, node, secs) in zip(buckets, results):
        code = str(int(Q[g0, S[0]]))
        tree[code] = node
        total += p0[S].sum() * c
    depths = evaluate_tree(first, tree)
    dist = {int(k): int(v) for k, v in zip(*np.unique(depths, return_counts=True))}
    print(f"{first.upper()}  objective under the prior {total:.4f}  | 2,315 answers: avg {depths.mean():.4f}  "
          f"over 6: {(depths > 6).sum()}  dist {dist}  {time.time()-t0:.0f}s", flush=True)
    json.dump(tree, open(os.path.join(RESULTS, f"tree_{first}.json"), "w"), separators=(",", ":"))
    save_json(f"tree_{first}_summary.json", {"opener": first, "lambda": LAMBDA, "objective": total,
                                             "avg": float(depths.mean()), "over6": int((depths > 6).sum()),
                                             "dist": dist, "seconds": round(time.time() - t0)})
