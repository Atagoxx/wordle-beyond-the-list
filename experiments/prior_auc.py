"""How well does each prior separate the 2,315 original answers from the other 10,657 accepted words?"""
import numpy as np

from common import IS_ANS, load_scores, rank, save_json


def auc(s):
    y = IS_ANS
    r = np.empty(len(s))
    r[np.argsort(np.asarray(s, float), kind="stable")] = np.arange(1, len(s) + 1)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


if __name__ == "__main__":
    s = {k: load_scores(k) for k in ["jev_plain", "jev_rules", "wordfreq_zipf", "jev_rules_wordle",
                                     "jev_explained_noul", "jev_explained_score"]}
    rows = [
        ("Jev plain: 'common everyday word' (no rules, no 'Wordle')", s["jev_plain"]),
        ("Jev + rules (no plurals / obscure words), no 'Wordle'", s["jev_rules"]),
        ("wordfreq Zipf frequency (free)", s["wordfreq_zipf"]),
        ("Jev + rules + 'could be a Wordle answer'", s["jev_rules_wordle"]),
        ("Jev, Wordle explained, Noul: 'on the solution list?'", s["jev_explained_noul"]),
        ("Jev, Wordle explained, 5-level Score", s["jev_explained_score"]),
        ("rank sum: Jev plain + wordfreq", rank(s["jev_plain"]) + rank(s["wordfreq_zipf"])),
        ("rank sum: Jev rules+'Wordle' + wordfreq", rank(s["jev_rules_wordle"]) + rank(s["wordfreq_zipf"])),
        ("rank sum: explained Score + rules+'Wordle' + wordfreq (tool)",
         rank(s["jev_explained_score"]) + rank(s["jev_rules_wordle"]) + rank(s["wordfreq_zipf"])),
    ]
    out = []
    for name, sc in rows:
        top = np.argsort(-np.asarray(sc, float), kind="stable")[:2315]
        a, hit = auc(sc), int(IS_ANS[top].sum())
        out.append({"prior": name, "auc": a, "answers_in_top_2315": hit})
        print(f"{name:<62} AUC {a:.4f}   true answers among its top 2,315: {hit}")
    save_json("prior_auc.json", out)
