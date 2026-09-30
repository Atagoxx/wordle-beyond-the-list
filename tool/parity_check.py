"""Check that the browser engine (tool/engine.js, run under node) agrees with the Python experiments:
same feedback codes and the same recommended guess, under both the base prior and a history-aware prior.

usage: python3 tool/parity_check.py      (needs node; the history-aware case also needs data/nyt_history.json)
"""
import json
import os
import random
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))
from common import G, INDEX, Q, choose, tool_prior  # noqa: E402


def python_side(w, histories):
    out = []
    for h in histories:
        S = np.flatnonzero(w > 0)
        for row in h:
            S = S[Q[INDEX[row["word"]], S] == row["code"]]
        out.append([len(S), G[choose(S, w)] if len(S) else None])
    return out


def js_side(w, histories, pairs):
    inp = json.dumps({"words": "".join(G), "weights": [float(x) for x in w], "histories": histories, "pairs": pairs})
    r = subprocess.run(["node", os.path.join(HERE, "parity.js")], input=inp, capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def game(answer, guesses):
    """Rows of a game (feedback computed from the answer) to use as a test state."""
    return [{"word": g, "code": int(Q[INDEX[g], INDEX[answer]])} for g in guesses]


if __name__ == "__main__":
    rng = random.Random(0)
    pairs = [[rng.choice(G), rng.choice(G)] for _ in range(5000)] + [["speed", "abide"], ["abbey", "keeps"]]
    histories = [[], game("elegy", ["tares"]), game("sloop", ["salet"]), game("copse", ["crane", "close"]),
                 game("elegy", ["salet", "colin"]), game("squid", ["tares", "colin"]), game("sloop", ["salet", "krill"])]
    priors = {"base prior": tool_prior()}
    if os.path.exists(os.path.join(os.path.dirname(HERE), "data", "nyt_history.json")):
        import backtest as bt
        priors["history-aware prior (2026-09-26)"] = bt.weights("P2s", "2026-09-26")
    ok = True
    for name, w in priors.items():
        # the empty history is slow on the Python side (12,972 candidates) but is the page's first screen
        py = python_side(w, histories)
        js = js_side(w, histories, pairs)
        codes_ok = all(c == int(Q[INDEX[g], INDEX[a]]) for c, (g, a) in zip(js["codes"], pairs))
        rec_ok = [list(a) == list(b) for a, b in zip(py, js["rec"])]
        ok &= codes_ok and all(rec_ok)
        print(f"{name}: {len(pairs)} feedback codes {'match' if codes_ok else 'DIFFER'}; recommendations "
              f"{sum(rec_ok)}/{len(rec_ok)} match: " + "; ".join(f"{p[0]} left -> {p[1]}" for p in py))
    sys.exit(0 if ok else 1)
