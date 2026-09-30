"""Same solver, different priors (no answer list). Each prior is rank-matched onto one value curve,
so only the ORDERING of words differs between priors. Combined priors add ranks.

The same 2,315-word list is used both to compare priors and to report the score, so the best row here is
slightly optimistic; the history backtest (backtest.py) is the out-of-sample check.
"""
import time

from common import evaluate, load_scores, matched, rank, save_json

if __name__ == "__main__":
    plain, rules, rw = load_scores("jev_plain"), load_scores("jev_rules"), load_scores("jev_rules_wordle")
    en, es, zf = load_scores("jev_explained_noul"), load_scores("jev_explained_score"), load_scores("wordfreq_zipf")
    runs = [
        ("Jev plain (no rules, no 'Wordle')", matched(plain)),
        ("Jev rules, no 'Wordle'", matched(rules)),
        ("wordfreq (free word-frequency list)", matched(zf)),
        ("Jev rules + 'Wordle'", matched(rw)),
        ("Jev rules, no 'Wordle' + wordfreq", matched(rank(rules) + rank(zf))),
        ("Jev rules + 'Wordle' + wordfreq", matched(rank(rw) + rank(zf))),
        ("Jev explained, Noul (on the list?)", matched(en)),
        ("Jev explained, 5-level Score", matched(es)),
        ("Jev explained Score, squared", matched(es, 2)),
        ("Jev explained Score + wordfreq", matched(rank(es) + rank(zf))),
        ("Jev explained Score + Jev rules+'Wordle' + wordfreq (tool)", matched(rank(es) + rank(rw) + rank(zf))),
    ]
    t = time.time()
    out = []
    for name, w in runs:
        avg, fails, dist, first = evaluate(w)
        out.append({"prior": name, "first": first, "avg": avg, "over6": fails, "dist": dist})
        print(f"{name:<58} first {first.upper()}  avg {avg:.4f}  over 6: {fails}  dist {dist}  ({time.time()-t:.0f}s)",
              flush=True)
    save_json("compare_priors.json", out)
