#!/usr/bin/env bash
# Free follow-up checks/pilots. Preserve archived results; no network or API calls.
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LAST_DAY=2026-09-25
out=cache/reproduced_followup
mkdir -p "$out"
mode=${1:-verify}
case "$mode" in
  verify)
    python3 experiments/verify_followup.py --output "$out/validation.json"
    ;;
  solver-checks)
    python3 experiments/verify_followup.py --solver-checks --output "$out/solver_validation.json"
    ;;
  pilots)
    python3 experiments/verify_followup.py --output "$out/validation.json"
    python3 experiments/pilot_branch_cost.py --limit 96 --seed 20260927 --workers 2 \
      --source-label 'Pinned original NYT history snapshot through 2026-09-25' \
      --output "$out/pilot_branch_cost_original_96.json"
    python3 experiments/pilot_rollout.py --limit 96 --seed 20260927 --workers 2 --mode second --top 6 \
      --source-label 'Pinned original NYT history snapshot through 2026-09-25' \
      --output "$out/pilot_rollout_second_original_96.json"
    python3 experiments/temporal_prior.py --output "$out/temporal_prior.json"
    python3 experiments/pilot_temporal.py --fits "$out/temporal_prior.json" \
      --limit 64 --seed 20260927 --workers 2 --output "$out/pilot_temporal_original_64.json"
    ;;
  certificate)
    python3 experiments/six_turn_guard_fast.py --seconds 120 --first salet --prior-day 2026-09-26 \
      --output "$out/six_turn_salet_fast_certificate.json"
    ;;
  *)
    echo 'Usage: bash reproduce_followup.sh [verify|solver-checks|pilots|certificate]' >&2
    exit 2
    ;;
esac
