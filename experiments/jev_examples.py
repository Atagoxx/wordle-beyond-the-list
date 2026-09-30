"""Call Jev with the article's prompts for a few example words and keep the raw requests and responses.

The key comes from the TYPESAFE_API_KEY environment variable and is never printed or saved.
usage: TYPESAFE_API_KEY=... python3 experiments/jev_examples.py results/jev_examples_<date>
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://api.typesafe.ai/v1/systemone"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

# Verbatim from jev_score_words.py and jev_player.py.
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
RULES_WORDLE = ("Wordle answers are common, everyday English words that most adult native speakers know (no plurals "
                "ending in s, no obscure, archaic, or technical words).")
TONES = "TONES: T=gray, O=yellow, N=yellow, E=gray, S=gray"

WORDS = ["crane", "cigar", "drama", "caput", "quoth", "pilow", "shool", "salet"]


def noul(text):
    return {"type": "noul", "instructions": text}


def score(text):
    return {"type": "score", "instructions": text, "criteria": LEVELS}


REQUESTS = {
    "1-plain": ({"context": "Judging English vocabulary."},
                {w: noul(f"'{w.upper()}' is a common, everyday English word that most adult native speakers know.") for w in WORDS}),
    "2-rules": ({"context": RULES},
                {w: noul(f"'{w.upper()}' is a common, everyday English word of the kind described.") for w in WORDS}),
    "3-rules-wordle": ({"context": RULES_WORDLE},
                       {w: noul(f"'{w.upper()}' is a common everyday English word that could be a Wordle answer.") for w in WORDS}),
    "4-explained-noul": ({"game": WORDLE_EXPLAINED},
                         {w: noul(f"'{w.upper()}' is on the official Wordle solution list (it can be a daily answer).") for w in WORDS}),
    "5-explained-score": ({"game": WORDLE_EXPLAINED},
                          {w: score(f"How likely is '{w.upper()}' to be on the Wordle solution list?") for w in WORDS}),
    # Second turn of the logged game whose answer was CRONY (AMONG, CRONY and FROND all still fit).
    "6-player": ({"game": WORDLE_EXPLAINED, "previous_guesses": [TONES],
                  "note": "Every word asked about below is still consistent with all the feedback so far."},
                 {w: noul(f"'{w.upper()}' is today's hidden Wordle answer.") for w in ["among", "crony", "frond"]}),
    # The colour check, one question per request as in the experiment: AMONG fits the feedback, CRANE does not.
    "7-probe-among": ({"game": WORDLE_EXPLAINED, "previous_guesses": [TONES]},
                      {"q": noul("'AMONG' is still consistent with all the feedback above, so it could be the hidden answer.")}),
    "8-probe-crane": ({"game": WORDLE_EXPLAINED, "previous_guesses": [TONES]},
                      {"q": noul("'CRANE' is still consistent with all the feedback above, so it could be the hidden answer.")}),
}

key = os.environ.get("TYPESAFE_API_KEY", "").strip()
if not key:
    sys.exit("TYPESAFE_API_KEY is not set")

total_in = 0
for name, (state, questions) in REQUESTS.items():
    body = {"model": "jev-latest", "state": state, "questions": questions}
    (OUT / f"{name}.request.json").write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            raw = res.read().decode()
    except urllib.error.HTTPError as err:
        print(f"{name}: HTTP {err.code} {err.read().decode()[:300]}")
        continue
    (OUT / f"{name}.response.json").write_text(raw + "\n")
    data = json.loads(raw)
    usage = data.get("usage", {})
    total_in += usage.get("input_tokens", 0)
    print(f"\n{name}: model {data.get('model')}, {time.time() - started:.1f}s, usage {usage}")
    for qid, answer in data.get("answers", {}).items():
        if answer.get("type") == "score":
            probs = " ".join(f"{float(p):.2f}" for p in answer.get("probabilities", {}).values())
            print(f"  {qid:6} score {answer.get('score')}  probabilities [{probs}]  confidence {answer.get('confidence')}")
        else:
            print(f"  {qid:6} noul {answer.get('noul')}")

print(f"\ninput tokens {total_in}, about ${total_in * 0.042 / 1e6:.5f} at $0.042 per million")
