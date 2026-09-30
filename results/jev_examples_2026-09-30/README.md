# Jev examples, September 30, 2026

The experiment saved one number per word for each prompt (`data/priors/`). To show everything Jev
returns, the same prompts were sent again on September 30, 2026 for eight example words, and the raw
requests and replies were kept here.

- Sent by `experiments/jev_examples.py` directly to `POST https://api.typesafe.ai/v1/systemone`, without
  the SDK. Each `*.request.json` is the exact request body and each `*.response.json` is the reply as
  received. The API key is read from the environment and appears in neither.
- Model: requested as `jev-latest`, answered by `jev-1.13.0`. 5,207 input tokens in total, about US$0.0002.
- Requests 1 to 5 are the five scoring prompts of `experiments/jev_score_words.py` (`jev_plain`,
  `jev_rules`, `jev_rules_wordle`, `jev_explained_noul`, `jev_explained_score`) for CRANE, CIGAR, DRAMA,
  CAPUT, QUOTH, PILOW, SHOOL and SALET. The Score replies include the per-level probabilities, the
  confidence and the level texts.
- Request 6 is the second turn of a logged Jev-as-player game whose answer was CRONY
  (`results/original_logs/jev_player.log`), asked for three words that still fit the feedback. AMONG gets
  the highest answer, the same second guess as in the log.
- Requests 7 and 8 are the colour-check question of `experiments/jev_player.py`, for a word that fits the
  feedback (AMONG, 0.66) and one that doesn't (CRANE, 0.21).
- Compared with the September scores in `data/priors/`, the Noul answers differ by at most 0.07 (mean
  0.015) and the Scores by at most 0.05 on the 0 to 4 scale. Jev's answers are not fully deterministic,
  and the model may have changed since September.

中文：实验时每个词只存了一个数。2026 年 9 月 30 日用同样的提示词为 8 个示例词重新调用了一次，这里保存完整的原始请求和返回（含 Score 每档的概率和置信度），模型版本 jev-1.13.0，与 9 月的结果相差在 0.07 以内。
