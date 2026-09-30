"""What does not knowing the answer list cost, and how much does a Jev prior win back?

All 12,972 accepted words are candidates. Same guess rule for every run; only the prior differs:
  - oracle: the solver knows the 2,315-word list (reference)
  - uniform: every accepted word equally likely
  - Jev p^a: Jev's probability that the word is an everyday word that could be a Wordle answer, sharpened by a
usage: python3 experiments/unknown_list.py [a ...]   (default 1 2 4)
"""
import sys
import time

import numpy as np

from common import N, evaluate, load_scores, save_json

if __name__ == "__main__":
    jp = load_scores("jev_rules_wordle")
    runs = [("oracle: knows the 2,315-word list", np.ones(N), True),
            ("no list: uniform over 12,972 words", np.ones(N), False)]
    for a in [float(x) for x in (sys.argv[1:] or ["1", "2", "4"])]:
        runs.append((f"no list: Jev prior p^{a:g}", np.maximum(jp, 1e-4) ** a, False))
    t = time.time()
    out = []
    for name, w, restrict in runs:
        avg, fails, dist, first = evaluate(w, restrict)
        out.append({"run": name, "first": first, "avg": avg, "over6": fails, "dist": dist})
        print(f"{name:<36} first {first.upper()}  avg {avg:.4f}  over 6: {fails}  dist {dist}  ({time.time()-t:.0f}s)",
              flush=True)
    save_json("unknown_list.json", out)
