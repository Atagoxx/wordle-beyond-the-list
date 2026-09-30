# Wordle, beyond the "optimal" solver

[中文说明](README.zh-CN.md)

September 27 follow-up: [report on the follow-up experiments (Chinese)](FOLLOWUP_REPORT.zh-CN.md).
Code, pinned raw inputs, per-day results, null results, and verification are archived; run
`bash reproduce_followup.sh`. The report corrects three older shorthand claims below: T0 is a heuristic
original-list baseline, not the proven optimal tree; P2 uses original-list membership as a category feature
without restricting candidates to that list; shortlist tree search is not exhaustive over all actions.

The textbook Wordle solver is provably optimal: open with SALET and you average 3.42 guesses. But that proof
assumes something that stopped being true. It assumes every answer comes from the original 2,315-word list,
with every word equally likely. In reality The New York Times now adds new words, and since February 2026 it
also repeats old answers.

This project builds a solver that never looks at the answer list. It combines:

- **a word prior from Jev**: TypeSafe's Jev, a "System One" model that answers typed questions with
  probabilities, scores how likely each of the 12,972 accepted words is to be an answer;
- **the real answer history**: 1,925 days from the NYT endpoint;
- **plain information-theoretic search**.

It was tested in a walk-forward backtest, where each day only sees the answers published before it:

| Solver, 2025-01-01 to 2026-09-25 (633 days) | Avg guesses | Days it cannot find the answer | Solved in ≤ 3 |
|---|---|---|---|
| Textbook optimum (original list, SALET) | 3.49* | **51** | 50.1% |
| Textbook, minus answers already used | 3.09* | **76** | 71.9% |
| No list, Jev + word-frequency prior (no history) | 3.58 | 0 | 48.0% |
| **No list, history-aware prior + Jev prior (this repo)** | **3.10** | **0** | **79.3%** |

\* Averaged only over the days the answer was still a candidate. On the other days the solver runs out of
words. None of the solvers ever needed more than 6 guesses.

On days both can solve, the history-aware solver needs 0.42 fewer guesses than the textbook optimum in 2025
(95% CI 0.35–0.49) and 0.50 fewer in 2026 (0.41–0.59). It also never misses an answer.

## What we learned

- **The model of reality beat the search.** Where the savings came from:

  | Source | Guesses saved |
  |---|---|
  | Using the real answer history | ~0.48 |
  | Jev's word prior (ablation) | ~0.06 |
  | Exhaustive decision-tree search instead of a one-step rule | ~0.01–0.02 |

- **Don't hard-code "it never happens".** "Answers never repeat" held for four and a half years, until
  2026-02-02, when CIGAR (the very first answer) came back. A solver that deleted used words missed 57 days in
  2026. Keeping a small, data-driven probability on every kind of word handled the change without missing a
  day.
- **Jev is a good prior and a bad player.** Jev playing on its own averaged 4.19 guesses, significantly
  worse than a solver with no prior at all. It also judged whether a word fits the colors correctly only 61% of
  the time. As a prior inside the solver it recovers about three quarters of the cost of not knowing the list.
  Scoring all 12,972 words costs 1.4–7.6 US cents per prompt.
- **Prompting matters, and so does contamination.** Asking Jev for a graded score with written criteria gave
  the best separation of answers from non-answers (AUC 0.976). Adding just the word "Wordle" to the prompt
  raised AUC from 0.911 to 0.961, which suggests Jev has seen the lists.
- **Two honest null results.** A better-calibrated prior (+0.22 bits of log-likelihood per day in 2026) and
  retuned search constants both left the guess count unchanged. See experiments 12–13.

The full log, with every experiment, number, caveat and failure, is in [EXPERIMENTS.md](EXPERIMENTS.md).

## The tool

`tool/` is a single-file web page. You type in each guess and tap the tiles to match the colors. It suggests
the next guess, shows the remaining candidates by probability, and records the day's answer when you solve it,
so its history stays current. Jev's scores are baked into the page ahead of time, so everything runs in the
browser: no API calls, no key. To re-score with your own key or try new prompts, run
`experiments/jev_score_words.py`, then `experiments/build_prior.py` and `tool/build.py`.

```bash
# The reported runs' pinned history is included; do not overwrite it before reproducing them.
python3 experiments/backtest.py P2s,P2,P0,T0,T1  # writes results/backtest_summary.json, used by the page
python3 tool/build.py                            # -> dist/wordle-solver.html
python3 tool/parity_check.py                     # the browser engine agrees with the Python code
```

The UI is currently in Chinese.

## Repository layout

```
experiments/   one script per experiment (numbered in EXPERIMENTS.md); common.py holds the solver core
data/          word lists and the per-word prior scores (see data/README.md for sources and licenses)
results/       logs and JSON from the re-run; original_logs/ holds the first runs verbatim;
               jev_examples_2026-09-30/ keeps raw Jev requests and replies for eight example words
tool/          engine.js (browser solver), template.html, build.py, parity_check.py
reproduce.sh   re-runs every deterministic experiment (~75 min on 4 cores)
```

Requirements: Python 3.10+ with numpy. Optional extras:

- `wordfreq`, to rebuild the frequency prior;
- `typesafe-sdk` and a `TYPESAFE_API_KEY`, for the Jev experiments;
- Node.js, for the parity check.
- `scipy`, for the follow-up temporal-prior fitting.

The feedback matrix (168 MB) is built on first use and cached in `cache/`.

## Data and licenses

- **Code** is MIT licensed ([LICENSE](LICENSE)).
- **Derived data** (Jev scores, word frequencies, the packed prior and the results) is CC BY-SA 4.0, because
  it includes data from [wordfreq](https://github.com/rspeer/wordfreq).
- The word lists come from the MIT-licensed npm package `wordle-words`.
- **The reported history snapshot is archived.** `data/nyt_history.json` is the unmodified snapshot used for every reported run;
  provenance and hashes are in `data/history_provenance.json`. Use this pinned input for reproduction.
  `experiments/fetch_history.py` can refresh a live tool separately. Built pages in `dist/` remain untracked.

Details are in [data/README.md](data/README.md).

## Disclosure

I am not affiliated with TypeSafe. I got API access to Jev through the public waitlist and paid for it
myself. All Jev calls in this project cost about US$0.25 in total. Wordle is a trademark of The New York
Times, and this project is not affiliated with it.
