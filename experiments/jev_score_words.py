"""Ask Jev (TypeSafe's "System One" model) to score every accepted Wordle guess. Writes data/priors/<variant>.json.

Needs `pip install typesafe-sdk` and TYPESAFE_API_KEY in the environment. Each variant scores 12,972 words in
5-11 s and cost $0.013-0.076 (input tokens at $0.042 per million, September 2026). Jev's answers are not fully
deterministic, so a re-run gives slightly different numbers than the files shipped in data/priors/.

usage: python3 experiments/jev_score_words.py <variant> [...]
variants: jev_plain jev_rules jev_rules_wordle jev_explained_noul jev_explained_score
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from typesafe_sdk import Noul, Score, TypeSafeClient

from common import G, PRIORS

WORDLE_EXPLAINED = (
    "Wordle is the New York Times daily word game. Players have six tries to guess a hidden five-letter word; "
    "after each guess every letter is marked green (right letter, right spot), yellow (in the word, wrong spot) "
    "or gray (not in the word). The game accepts about 13,000 five-letter words as guesses, but the hidden answer "
    "is always drawn from a separate, hand-curated list of about 2,300 solutions picked by creator Josh Wardle and "
    "later edited by the New York Times. Solutions are familiar words a typical adult knows: plurals ending in S and "
    "most past tenses ending in ED are excluded, as are obscure, archaic, technical, foreign, and offensive words. "
    "The other ~10,000 accepted guesses are valid Scrabble-style words that are never solutions.")
LEVELS = ["Never a Wordle solution (obscure, archaic, technical, a plural or inflection, or a misspelling)",
          "Unlikely: a real word but too uncommon or too derivative",
          "Borderline: known to many people but not typical",
          "Likely: a familiar word that fits the solution list",
          "Almost certainly on the Wordle solution list: a typical everyday word"]
RULES = ("We want common, everyday English words that most adult native speakers know: no plurals ending in s, "
         "no obscure, archaic, or technical words.")

# variant -> (state, question builder, batch size)
VARIANTS = {
    "jev_plain": ({"context": "Judging English vocabulary."}, lambda w: Noul(
        instructions=f"'{w.upper()}' is a common, everyday English word that most adult native speakers know."), 200),
    "jev_rules": ({"context": RULES}, lambda w: Noul(
        instructions=f"'{w.upper()}' is a common, everyday English word of the kind described."), 200),
    "jev_rules_wordle": ({"context": "Wordle answers are common, everyday English words that most adult native "
                                     "speakers know (no plurals ending in s, no obscure, archaic, or technical words)."},
                         lambda w: Noul(instructions=f"'{w.upper()}' is a common everyday English word that could be "
                                                     f"a Wordle answer."), 200),
    "jev_explained_noul": ({"game": WORDLE_EXPLAINED}, lambda w: Noul(
        instructions=f"'{w.upper()}' is on the official Wordle solution list (it can be a daily answer)."), 200),
    "jev_explained_score": ({"game": WORDLE_EXPLAINED}, lambda w: Score(
        instructions=f"How likely is '{w.upper()}' to be on the Wordle solution list?", criteria=LEVELS), 100),
}


def score(variant):
    state, question, size = VARIANTS[variant]
    usage = [0]

    def run(batch):
        with TypeSafeClient() as c:
            r = c.system_one(state=state, questions={w: question(w) for w in batch})
        usage[0] += r.usage.input_tokens
        if isinstance(question(batch[0]), Score):
            return {w: r.scores[w].score for w in batch}
        return {w: r.nouls[w].noul for w in batch}

    t, out = time.time(), {}
    with ThreadPoolExecutor(8) as ex:
        for res in ex.map(run, [G[i:i + size] for i in range(0, len(G), size)]):
            out.update(res)
    json.dump(out, open(os.path.join(PRIORS, variant + ".json"), "w"))
    print(f"{variant}: scored {len(out)} words in {time.time()-t:.0f}s, input tokens {usage[0]}, "
          f"cost ${usage[0]*0.042/1e6:.4f}", flush=True)


if __name__ == "__main__":
    for v in sys.argv[1:] or ["-h"]:
        if v not in VARIANTS:
            sys.exit(__doc__)
        score(v)
