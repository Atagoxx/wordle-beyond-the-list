# Experiment log

This file lists every experiment behind this project in the order it was run (September 2026), including the
ones that failed or found nothing. For each one it gives the question, the method, the numbers and what we
concluded. Scripts live in `experiments/`, and output from the re-run lives in `results/`.

**September 27 follow-up archive:** see [FOLLOWUP_REPORT.zh-CN.md](FOLLOWUP_REPORT.zh-CN.md) for the new
paired pilots, input provenance, Jev attribution, and publication caveats. It is an experiment handoff,
not a duplicate blog. `bash reproduce_followup.sh` verifies saved evidence; add `pilots` to rerun the free
small-sample experiments into `cache/reproduced_followup/` without overwriting the archived outputs.
Important corrections to older wording below: T0 is the one-step heuristic with original-list support,
not the proven optimum; P2 uses list membership as a feature; experiment 7 restricts candidate actions;
its 0.01–0.02 gain is not an upper bound for deeper search under P2. Historical walk-forward evaluation
also does not establish an untouched holdout or exclude offline model training contamination.

Conventions:

- "Guesses" means the average number of guesses to solve, counting the winning guess. Lower is better.
- "Over 6" counts games that would have been lost in the real game.
- Runs over "all 2,315 answers" play every word on the original answer list once. The solver is
  deterministic, so these are exact values, not samples.
- Feedback codes are `sum(color_i * 3**i)`, with gray 0, yellow 1 and green 2. Duplicate letters use the
  standard left-to-right remaining-count rule.
- Unless noted, the guess rule is the one-step rule in `common.choose()`:
  `cost = 1 + (1 - P(guess is the answer)) * (1 + 0.55*rem + 0.1*rem^2)`, where `rem` is the entropy left after
  the guess, in bits. The constants were set by hand. Experiment 13 re-tunes them.

Reproducibility: `./reproduce.sh` re-runs every deterministic experiment (about 75 minutes on 4 cores). The
re-run matched the original run exactly, down to each of the 633 backtest days. Two things are not re-run: the
paid Jev experiments (3, 5) and the NYT download (8). Their original logs are in `results/original_logs/`.

---

## 1. Textbook setting: the solver knows the 2,315-word answer list

`experiments/known_list.py bench`

| Strategy | Opener | Guesses | Worst | Over 6 |
|---|---|---|---|---|
| Naive filter (guess the first word still possible) | CRANE | 3.9490 | 9 | 44 |
| Greedy entropy | SOARE (its own pick) | 3.4644 | 6 | 0 |
| Greedy entropy | CRANE | 3.4384 | 6 | 0 |
| Entropy + chance of winning outright | CRANE | 3.4389 | 6 | 0 |
| Lookahead search (exact expectation over the top 12 guesses per node) | CRANE | **3.4255** | 5 | 0 |

- Published exhaustive searches give the optimum for this list: SALET, 3.4212 guesses (7,920 in total). To
  our knowledge the published optimum for a CRANE opener is 7,930 total = 3.4255, which our lookahead matches.
  Please double-check this against the published tables.
- Lower bound: TRACE splits the list into the most groups (150), so at most about 150 answers can be solved by
  guess 2. That gives an average of at least (1 + 149·2 + 2,165·3) / 2,315 ≈ 2.93. Claims of "2-point-something
  on average" are luck or leakage.
- The highest-information opener (SOARE) is not the best opener. CRANE beats it by 0.026 guesses. Greedy
  information is not the same as the best whole-game result.
- A lookahead run with SALET was stopped by the 25-minute limit. The published value covers it.

## 2. Without the answer list: what it costs, and what a Jev prior wins back

`experiments/unknown_list.py`. All 12,972 accepted words are candidates. Only the prior over candidates
changes.

| Prior | Opener (own pick) | Guesses | Over 6 |
|---|---|---|---|
| Oracle: knows the 2,315 list (same guess rule) | SOARE | 3.4635 | 0 |
| No list, every accepted word equally likely | TARES | 3.9620 | 2 |
| No list, Jev prior p | TARES | 3.5987 | 1 |
| No list, Jev prior p² | TARES | 3.5866 | 3 |
| No list, Jev prior p⁴ | TRACE | 3.5594 | 7 |

- Not knowing the list costs half a guess (3.46 → 3.96). One cheap Jev prior wins back about 73% of that.
- Sharper priors lower the average but lose more games. Jev scores some real answers very low (CAPUT, TWIXT and
  QUOTH get 0.02–0.05), and sharpening pushes them to the very end. It also scores some misspellings and rare
  words (PILOW, SHOOL) above 0.9.
- The Jev prior here is `jev_rules_wordle`. The prompt describes the rules (no plurals, no obscure words) and
  says "could be a Wordle answer". Scoring all 12,972 words took 6 s and cost $0.015.

## 3. Which prompt makes a good prior? Separating answers from non-answers

`experiments/jev_score_words.py` generates the scores (paid). `experiments/prior_auc.py` produces this table.
AUC is the probability that a random answer scores above a random non-answer.

| Prior | AUC | Answers among its top 2,315 | Cost |
|---|---|---|---|
| Jev plain: "a common everyday word" (no rules, no "Wordle") | 0.8777 | 1,289 | $0.0145 |
| Jev + rules (no plurals / obscure words), no "Wordle" | 0.9114 | 1,662 | $0.0135 |
| wordfreq Zipf frequency (free, no model) | 0.9219 | 1,451 | $0 |
| Jev + rules + "could be a Wordle answer" | 0.9614 | 1,813 | $0.0151 |
| Jev, Wordle explained in detail, Noul: "is it on the solution list?" | 0.9299 | 1,586 | $0.0166 |
| **Jev, Wordle explained in detail, 5-level Score with written criteria** | **0.9761** | **1,909** | $0.0756 |
| Rank sum: explained Score + rules+"Wordle" + wordfreq (the tool's prior) | 0.9790 | 1,935 | — |

- The rules in the prompt help (0.878 → 0.911).
- Just adding the word "Wordle" helps a lot more (0.911 → 0.961). This is a contamination warning: Jev has
  probably seen Wordle answer lists in training. Without "Wordle" in the prompt, Jev is worse than a free
  word-frequency list.
- Asking directly about list membership (Noul, 0.930) is worse than asking for a graded Score with explicit
  criteria (0.976). Jev does better with concrete criteria than with "do you remember the list".
- Prompts and code are in `experiments/jev_score_words.py`. The Noul and Score questions share one
  detailed explanation of Wordle.

## 4. Priors inside the solver (rank-matched)

`experiments/compare_priors.py`. Each prior is mapped onto the same probability curve by rank, so only the
ordering differs. Combined priors add ranks.

| Prior (no list) | Opener | Guesses | Over 6 |
|---|---|---|---|
| Jev explained Noul ("on the list?") * | TARES | 3.6060 | 2 |
| Jev plain, no rules, no "Wordle" * | TARES | 3.6354 | 1 |
| Jev rules, no "Wordle" | TARES | 3.6436 | 1 |
| wordfreq | TARES | 3.6168 | 4 |
| Jev rules + "Wordle" | TARES | 3.5987 | 1 |
| Jev rules, no "Wordle" + wordfreq | TARES | 3.5991 | 1 |
| Jev rules + "Wordle" + wordfreq | TARES | 3.5680 | 0 |
| Jev explained Score | TARES | 3.5806 | 0 |
| Jev explained Score, squared | SOARE | 3.5784 | 6 |
| Jev explained Score + wordfreq | TARES | 3.5754 | 0 |
| **Jev explained Score + Jev rules+"Wordle" + wordfreq** (shipped in the tool) | TARES | **3.5646** | 0 |

\* Rows added in the re-run. They were only AUC-tested originally.

- AUC does not translate one-to-one into guesses. Jev plain has the lowest AUC (0.878) but beats "Jev rules,
  no Wordle" (AUC 0.911) inside the solver. What matters is the ordering near the top, not overall
  separation.

- Two mediocre priors that make different mistakes combine into a good one. Jev without "Wordle" (3.64) plus
  wordfreq (3.62) gives 3.60, so the prompt doesn't need the contaminating word.
- **Caveat:** the prior was chosen on the same 2,315 answers used to report its score, so 3.5646 is slightly
  optimistic. The combinations differ by only 0.01–0.03, so the optimism is small. The history backtest
  (experiment 9) is the clean out-of-sample test.

## 5. Jev as a player, versus Jev as a prior

`experiments/jev_player.py` (paid, about $0.11). Jev is given the rules plus every earlier guess with its
colors. Each turn it scores every word still consistent with the feedback, and the top-scored word is guessed.
Code only enforces the color constraints.

**Can Jev check the constraints itself?** In 300 items, half consistent and half not, Jev judged whether a word
fits one row of feedback. Accuracy was **61.0%** (chance is 50%) with AUC 0.732. Letter-by-letter checking is
not something it does reliably, which is expected for a token-based model. So code must do the filtering.

**200 sampled answers** (fixed seed, same 200 for every comparison):

| Compared with (same 200 games) | Their avg | Jev player minus them | 95% CI (bootstrap) | t | Jev fewer / same / more |
|---|---|---|---|---|---|
| Solver, uniform prior (no list) | 3.915 | +0.275 | [+0.110, +0.445] | 3.22 | 43 / 77 / 80 |
| Solver + static Jev prior (no list) | 3.605 | +0.585 | [+0.425, +0.745] | 7.26 | 27 / 75 / 98 |
| Solver that knows the list | 3.460 | +0.730 | [+0.560, +0.900] | 8.36 | 29 / 53 / 118 |

- Jev playing on its own averaged **4.19** guesses and lost 6 of 200 games. That is worse than even the
  no-prior solver. It always guesses the word most likely to be the answer. It never picks a probe word that
  splits the candidates.
- The same Jev, used as a prior inside the solver, is the best option available without the list.
- Jev's answers are not deterministic. Games that started at the same time, before the cache was filled, each
  asked for the first move and got different openers (TONES, WHITE, NOISE, LAPSE, STORE).
- 2,665,961 input tokens, $0.112.

## 6. The first web tool, and which opener to use

The first tool shipped the rank-sum prior from experiment 4 with the one-step rule running in the browser. On
all 2,315 answers it averaged 3.5646, with 0 games over 6.

`experiments/openers.py`: fix the opener, then follow the tool for the rest of the game.

| Opener | Guesses | Solved in ≤ 3 | Over 6 |
|---|---|---|---|
| REAST | 3.5443 | 49.3% | 1 |
| SALET | 3.5495 | 48.7% | 2 |
| SLATE | 3.5516 | 49.0% | 2 |
| TRACE | 3.5516 | 49.3% | 2 |
| CRATE | 3.5594 | 48.4% | 2 |
| CRANE | 3.5603 | 48.0% | 1 |
| TARES (the rule's own pick) | 3.5641 | 47.9% | 0 |
| RATES | 3.5737 | 47.2% | 0 |
| RAISE | 3.5857 | 47.1% | 3 |
| SOARE | 3.5888 | 46.9% | 2 |
| ROATE | 3.5892 | 46.1% | 2 |
| LARES | 3.5905 | 46.6% | 2 |

The good openers are within about 0.02 guesses of each other. The ranking here was measured on the answer list
itself.

## 7. Near-exhaustive decision-tree search (no list)

`experiments/tree_search.py`. The objective at a node reached before guess d is
`E[guesses] + 5 · P(solved after guess 6)`, with probabilities taken from the prior over all 12,972 words.
Every guess tried is followed exactly through all of its feedback branches, with lower-bound pruning and
memoisation. Only the set of guesses tried per node is limited: the top 4/6/10/16 by the one-step rule,
depending on node size, plus the two most likely candidates.

| Opener | Objective under the prior (used to choose) | Guesses on the 2,315 answers | Over 6 | Time |
|---|---|---|---|---|
| REAST | 3.7743 | 3.5352 | 0 | 814 s |
| TARES | **3.7410** | 3.5469 | 0 | 649 s |
| SALET | 3.7471 | **3.5309** | 0 | 565 s |

- Full search gains only about 0.01–0.02 guesses over the one-step rule (REAST 3.544 → 3.535), but it removes
  the lost games.
- The rule was fixed in advance: choose the opener by the prior objective, never by the answer list. That rule
  picks TARES. The answer list would have picked SALET. The two measures disagree, and the gap is noise-sized.
- This version of the tool shipped the three trees. They were dropped later (experiment 10), because they were
  built for the old prior.

## 8. The real answer history

`experiments/fetch_history.py` and `experiments/history_stats.py`. The data comes from the endpoint the game
itself uses (`/svc/wordle/v2/<date>.json`): 1,925 days from 2021-06-19 to 2026-09-25, with none missing.

| Year | Days | Answers outside the original list | Repeats of an earlier answer |
|---|---|---|---|
| 2021 | 196 | 0 | 0 |
| 2022 | 365 | 0 | 0 |
| 2023 | 365 | 4 (GUANO, SNAFU, BALSA, KAZOO) | 0 |
| 2024 | 366 | 6 (LASER, PIOUS, BEAUT, MOMMY, PRIMP, UVULA) | 0 |
| 2025 | 365 | 19 (ATLAS, SQUID, INDIE, SPATE, SUEDE, …) | 0 |
| 2026 (to 09-25) | 268 | 32 (SITAR, OOMPH, MOOCH, SHRED, HYDRA, …) | **25** |

- Repeats started on **2026-02-02** with CIGAR, the very first Wordle answer. There have been 25 repeats in
  2026 so far: 21 were first used in 2021, 2 in 2022, and 1 each in 2023 and 2024. The gap between uses is
  1.4–5.1 years (median 4.6).
- All 61 answers from outside the list are still in the 12,972-word accepted list.
- The answers followed the original list's order until 2022-02-15 (day 241: AGORA was replaced by AROMA). The
  API's `editor` field appears from 2022-11-07.
- 476 words on the original list have never been used.
- We did not look for an official NYT announcement about repeats. Worth checking before publishing.

## 9. Walk-forward backtest: beating the "mathematical optimum" on real answers

`experiments/backtest.py`. Each day is played using only answers published before that day.

| Variant | Prior | Opener |
|---|---|---|
| T0 | Textbook: uniform over the original 2,315 list | SALET |
| T1 | Textbook minus words already used | SALET |
| P0 | The tool's base prior over all 12,972 words (no history) | TARES |
| P2 | History-aware (see below) | TARES |
| P2s | History-aware | SALET |

The history-aware prior works like this:

- Candidates fall into three groups: unused original words, already-used words, and words outside the original
  list.
- Each group's share is estimated from the previous 180 answers, with add-one smoothing.
- Inside a group, words are weighted by the base prior.
- No word is ever excluded.

| Variant | 2025 avg | 2025 not found | 2026 avg | 2026 not found | All 633 days | Solved in ≤ 3 |
|---|---|---|---|---|---|---|
| T0, "mathematical optimum" | 3.451* | 19 | 3.534* | 32 | 3.485*, **51 not found** | 50.1% |
| T1, textbook minus used words | 3.133* | 19 | 3.019* | 57 | 3.090*, **76 not found** | 71.9% |
| P0, tool without history | 3.540 | 0 | 3.642 | 0 | 3.583 | 48.0% |
| P2, history-aware + TARES | 3.134 | 0 | 3.235 | 0 | 3.177 | 76.9% |
| **P2s, history-aware + SALET** | **3.071** | **0** | **3.134** | **0** | **3.098** | **79.3%** |

\* Averaged only over days on which the answer was still a candidate. On the other days the solver runs out of
candidates and has no move. No variant ever needed more than 6 guesses.

Paired differences, on days both variants could solve, with normal 95% CIs:

| Comparison | 2025 | 2026 |
|---|---|---|
| P2s − T0 | −0.422 [−0.493, −0.351] (346 days) | −0.496 [−0.586, −0.406] (236 days) |
| P2 − T0 | −0.361 [−0.439, −0.283] | −0.407 [−0.506, −0.307] |
| P0 − T0 | +0.090 [+0.005, +0.174] | +0.093 [+0.006, +0.180] |
| T1 − T0 | −0.318 [−0.380, −0.256] | −0.526 [−0.619, −0.434] (211 days) |

- The "optimal" solver is optimal only under its own assumption: a fixed 2,315-word list with every word
  equally likely. Real answers break that assumption. On 51 of 633 days it cannot find the answer at all, and on
  the other days the history-aware solver needs about 0.4–0.5 fewer guesses.
- Hard-coding "answers never repeat" (T1) worked until February 2026 and then failed on every repeat. Keeping
  a small, data-driven probability on every group handled the change without missing a day.
- In the final 180-day window (as of 2026-09-26) the group shares were about 75% unused original, 11% repeats
  and 14% outside the list.
- **Caveat:** SALET vs TARES for P2 was decided after seeing both backtests (a two-way choice). Experiment 14
  chooses the opener without looking at the answers.

## 10. The history-aware tool

The web tool (`tool/`) was rebuilt around P2s:

- The tool computes the category prior itself, from the embedded history plus answers the user solves later.
- Solving a puzzle (an all-green row) adds that day's answer to the stored history, and the user can undo it.
- The opener is SALET.
- The engine is `tool/engine.js`. `tool/parity_check.py` checks that it agrees with the Python code on feedback
  codes and recommendations, under both priors.

An example of why the tool sometimes recommends a word that can't be the answer: on 2026-09-27, SALET left 73
candidates. 85% of the probability sat on 7 of them: SLIMY, SLURP, SLINK, SLYLY, SLUNK, SLOOP, SKULK. The tool
suggested KRILL, which is not a candidate. KRILL gives each of the 7 a different color pattern, and its
expected total is 3.13 guesses, against 3.16 for guessing SLIMY directly. The game was solved in 3
(SALET → KRILL → SLOOP).

## 11. Ablation: does Jev still matter once the history is used?

`experiments/ablation.py`. The history groups and the SALET opener stay fixed. Only the within-group word
score is swapped. All 633 days.

| Word score inside each group | Guesses | vs current (95% CI) | Solved in ≤ 3 | Over 6 |
|---|---|---|---|---|
| **Current: Jev ×2 + wordfreq** | **3.098** | — | 79.3% | 0 |
| Jev only (drop wordfreq) | 3.104 | +0.006 (±0.021), n.s. | 78.8% | 0 |
| wordfreq only (drop Jev) | 3.155 | **+0.057 (±0.030)** | 74.9% | 0 |
| No score (uniform inside groups) | 3.216 | +0.118 (±0.038) | 74.7% | 1 |

Rough attribution, in guesses saved:

| Source | Guesses saved |
|---|---|
| Real answer history | about 0.48 (P0 3.58 → P2s 3.10) |
| Jev | about 0.06 |
| wordfreq on top of Jev | about 0 |
| Full-tree search instead of the one-step rule | 0.01–0.02 (experiment 7) |

## 12. Attempted improvement A: a calibrated prior inside each group (P3), null result

`experiments/improve_prior.py`. P2 uses the base prior unchanged inside each group. P3 fits its shape
walk-forward, only from answers before the day being played:

- unused original words: `w ∝ base^a_U`
- words outside the list: `w ∝ base^a_X`
- repeats: `w ∝ base^a_R · exp(b · years since first use)`

The fits are MAP estimates on a grid, with weak priors a ~ N(1, 1) and b ~ N(0, 1). The repeat-age term was
added *after* seeing experiment 8, which is disclosed here, but b is only ever learned from earlier repeats.
P3w365 fits only on the last 365 days.

Fitted values settle at about a_U ≈ 1.0 (the base prior is already calibrated there), a_X ≈ 2 (new words are
the most familiar ones), a_R ≈ 2 and b ≈ 1.5 per year (older answers come back first).

| | 2025 | 2026 |
|---|---|---|
| log₂ P(real answer), P3 − P2 | +0.021 ± 0.020 bits | **+0.217 ± 0.097 bits** |
| Guesses, P3 − P2s | +0.000 (95% CI ±0.019) | +0.004 (±0.047) |
| Guesses, P3w365 − P2s | −0.003 (±0.026) | +0.015 (±0.048) |

By group in 2026, P3 vs P2s:

| Group | Days | P2s | P3 |
|---|---|---|---|
| Repeats | 25 | 3.88 | 3.84 |
| Outside the list | 32 | 3.84 | 3.75 |
| Unused original | 211 | 2.94 | 2.96 |

P3 also has one game over 6. Conclusion: the prior became measurably better at predicting the answer, but the
solver did not guess fewer times. Mass moved toward the rarer groups, and that made the common case slightly
worse. Not adopted.

## 13. Attempted improvement B: re-tuning the guess rule's constants, null result

`experiments/tune_cost.py`. A grid of 20 values for (c1, c2) in `1 + (1 − P(win))(1 + c1·rem + c2·rem²)` was
run with P2 and the SALET opener, tuned on 2025 only.

- The 2025 averages ranged only from 3.058 to 3.071; the default (0.55, 0.1) scored 3.071.
- The best pair, (0.35, 0.05), scored 3.138 on held-out 2026, versus 3.134 for the default: +0.004 ± 0.026.

The rule is on a plateau, so the remaining gap is not in these constants. Not adopted.

## 14. Choosing the opener without looking at the answers

`experiments/opener_history.py`. For each of 12 openers, the expected number of guesses was computed exactly
under the P2 prior of 2025-01-01. This walks the solver's full decision tree over every candidate, weighted by
prior mass, and uses no answers. That is then compared with the real walk-forward backtest.

| Opener | Expected under the prior (no answers used) | Real: all 633 days | 2025 | 2026 | Over 6 |
|---|---|---|---|---|---|
| **SLATE** | **3.1545** | **3.087** | 3.044 | 3.146 | 1 |
| REAST | 3.1557 | 3.130 | 3.088 | 3.187 | 0 |
| SALET (shipped) | 3.1568 | 3.098 | 3.071 | 3.134 | 0 |
| RAISE | 3.1804 | 3.139 | 3.129 | 3.153 | 2 |
| CRANE | 3.1814 | 3.137 | 3.112 | 3.172 | 1 |
| CRATE | 3.1858 | 3.123 | 3.101 | 3.153 | 2 |
| TRACE | 3.1862 | 3.117 | 3.074 | 3.175 | 1 |
| RATES | 3.1918 | 3.175 | 3.140 | 3.224 | 0 |
| SOARE | 3.1923 | 3.142 | 3.148 | 3.134 | 0 |
| ROATE | 3.1947 | 3.133 | 3.093 | 3.187 | 2 |
| TARES | 3.1954 | 3.177 | 3.134 | 3.235 | 0 |
| LARES | 3.2128 | 3.141 | 3.107 | 3.187 | 0 |

Paired against SALET over the 633 days:

| Opener | Difference | Fewer / same / more |
|---|---|---|
| SLATE | −0.011 ± 0.040 | 80 / 481 / 72 |
| REAST | +0.032 ± 0.050 | 105 / 410 / 118 |
| TARES | +0.079 ± 0.045 | 76 / 435 / 122 |

- A rule fixed in advance ("take the opener with the lowest expected guesses under the prior") would have
  picked SLATE. SLATE does as well as SALET on the real answers (3.087 vs 3.098, not significant). So the
  shipped result is not an artifact of choosing SALET after seeing the backtest.
- Under the prior, SLATE, REAST and SALET are tied within 0.002. The real results separate them by up to 0.04,
  which is within noise. The rank correlation between the prior estimate and the real result is 0.70.
- The prior's own estimate (3.15–3.21) is close to the real average (3.09–3.18), so it is roughly calibrated
  and slightly pessimistic.
- The tool keeps SALET. Switching to SLATE is not justified by these numbers, and SLATE has one game over 6.

---

## Cost

All Jev calls together cost about $0.25:

| Run | Cost |
|---|---|
| Five 12,972-word scoring runs | $0.015 + 0.0145 + 0.0135 + 0.0166 + 0.0756 |
| 200-game player and constraint probe | $0.112 |
| A few smoke tests | small amount |

Prices were $0.042 per million input tokens, with output free (TypeSafe, September 2026). The author has no
affiliation with TypeSafe and got API access through the public waitlist.

## Where improvement might still come from

The first place to look is where the remaining guesses go. In 2026 the history-aware solver averages:

| Kind of day | Share of days | Avg guesses |
|---|---|---|
| Unused original word | 79% | 2.94 |
| Repeat | 9% | 3.88 |
| Word outside the original list | 12% | 3.84 |

The common case is close to what any strategy can do with about 500 plausible candidates. The headroom is in
the rare kinds of day, where the answer has low prior probability. Ideas, roughly by expected value:

1. **A better model of which new words the editor adds.** Experiment 12 shows that sharpening alone moves
   probability around without saving guesses. Something that separates plausible additions from the other
   10,000 accepted words would. Candidate signals: frequency trends, word class, a Jev prompt aimed at
   "words an editor would add".
2. **Search that is aware of the prior on those days.** The one-step rule is tuned for the common case. Near-exact
   tree search per day (as in experiment 7, but under the history-aware prior) is expensive, but it is the
   honest way to measure how far the rule is from optimal. Earlier evidence puts the gain at ≤ 0.02.
3. **Better repeat modelling once more repeats exist.** With 25 so far, the age effect is clear, but its shape
   isn't.
4. **Averaging several Jev calls** to reduce its run-to-run noise.
5. **Product features rather than score:** hard mode (only legal guesses), an English UI, a random-opener
   option.

## Known limitations

- Experiments 2–7 report scores on the same 2,315-word list that was used to design the priors. Experiment 9
  is the out-of-sample test.
- The history-aware design (three groups, the 180-day window) was chosen knowing that the list had been
  extended and, later, that repeats existed. Only the parameters are fit walk-forward.
- "Not found" days count as failures for T0 and T1. A human would fall back to guessing freely.
- Jev may have memorised Wordle lists (experiment 3). Its outputs vary between identical calls.
- Hard mode is not modelled. The recommended probe words can be illegal in hard mode.
- The NYT endpoint is undocumented and can change.
