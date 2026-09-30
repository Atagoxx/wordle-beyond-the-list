#!/usr/bin/env bash
# Re-run every deterministic experiment and write results/*.log + results/*.json (about 75 min on 4 cores).
# Not included: the paid Jev calls (jev_score_words.py, jev_player.py) and the NYT download (fetch_history.py).
set -euo pipefail
cd "$(dirname "$0")"
run() { local name=$1; shift; echo "== $name: $*"; python3 "$@" 2>&1 | tee "results/$name.log"; }
mkdir -p results
run prior_auc          experiments/prior_auc.py
run history_stats      experiments/history_stats.py
run unknown_list       experiments/unknown_list.py
run compare_priors     experiments/compare_priors.py
run openers            experiments/openers.py
run backtest           experiments/backtest.py T0,T1,P0,P2,P2s
run ablation           experiments/ablation.py
for w in reast tares salet; do run "tree_$w" experiments/tree_search.py "$w"; done
STARTS=crane run known_list experiments/known_list.py bench
