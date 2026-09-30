"""Build the tool's base prior (data/tool_prior.json): rank-sum of Jev's explained 5-level Score, Jev's
rules+'Wordle' probability and wordfreq, mapped onto the common value curve, then quantized to 2 base-36 chars
per word (v = round(-100 * log10(w)))."""
import json
import os

import numpy as np

from common import DATA, G, N, evaluate, load_scores, matched, rank

if __name__ == "__main__":
    w = matched(rank(load_scores("jev_explained_score")) + rank(load_scores("jev_rules_wordle")) +
                rank(load_scores("wordfreq_zipf")))
    v = np.clip(np.round(-np.log10(w) * 100), 0, 1295).astype(int)
    wq = 10 ** (-v / 100)  # exactly what the page decodes
    weight_str = "".join(np.base_repr(x, 36).lower().rjust(2, "0") for x in v)
    word_str = "".join(G)
    assert len(weight_str) == 2 * N and len(word_str) == 5 * N
    json.dump({"words": word_str, "weights": weight_str}, open(os.path.join(DATA, "tool_prior.json"), "w"))
    avg, fails, dist, first = evaluate(wq)
    print(f"wrote data/tool_prior.json; on the 2,315 original answers: first {first.upper()} avg {avg:.4f} "
          f"over 6: {fails} {dist}")
