"""Bounded-width TRUE-certificate search; incomplete search returns UNKNOWN.

This complements six_turn_guard.py without changing its exhaustive semantics.
Only fully checked trees or exact two-turn/bound impossibility results enter the
shared proof cache. Exhausting a shortlist or local work budget is NOT FALSE.
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np

from common import ALL_GREEN, DATA, G, INDEX, N, Q, RESULTS, choose
from six_turn_guard import SixTurnGuard, verify_certificate, _BudgetExceeded, _state


class FastCertificateSearch(SixTurnGuard):
    def __init__(self, *, weights=None, max_seconds=120):
        super().__init__(weights=weights, max_nodes=None, max_guesses=None,
                         max_seconds=max_seconds)
        self.fast_probes = {}
        self.shortlist_unknown = set()

    def _probes(self, S):
        key = S.tobytes()
        if key in self.fast_probes:
            return self.fast_probes[key]
        largest = np.empty(N, dtype=np.int32)
        squared = np.empty(N, dtype=np.int64)
        self.stats["probe_scoring_calls"] += 1
        for start in range(0, N, 256):
            self._check_budget()
            stop = min(N, start + 256)
            rows = stop - start
            bins = Q[start:stop][:, S].astype(np.int64)
            bins += (np.arange(rows) * 243)[:, None]
            counts = np.bincount(bins.ravel(), minlength=rows * 243).reshape(rows, 243)
            counts[:, ALL_GREEN] = 0
            largest[start:stop] = counts.max(axis=1)
            squared[start:stop] = (counts * counts).sum(axis=1)
            self.stats["probe_scoring_rows"] += rows
        ranked = np.lexsort((np.arange(N), squared, largest))
        # Distinct partitions, even if different color-code names label them.
        selected, signatures = [], set()
        for candidate in ranked:
            self._check_budget()
            row = Q[int(candidate), S]
            labels, first, inverse = np.unique(row, return_index=True, return_inverse=True)
            canonical = np.argsort(np.argsort(first))[inverse].astype(np.int16)
            canonical[row == ALL_GREEN] = 243  # terminal is not an ordinary bucket
            signature = canonical.tobytes()
            if signature in signatures:
                continue
            signatures.add(signature)
            selected.append(int(candidate))
            if len(selected) == 16:
                break
        self.fast_probes[key] = selected
        return selected

    def _fast(self, S, remaining, width, allowance):
        self._check_budget()
        key = (S.tobytes(), remaining)
        if key in self.memo:
            self.stats["memo_hits"] += 1
            ok, policy = self.memo[key]
            return ("TRUE" if ok else "FALSE"), policy
        if allowance[0] <= 0:
            self.stats["local_budget_unknown"] += 1
            return "UNKNOWN", None
        allowance[0] -= 1
        if len(S) <= remaining or remaining <= 2 or len(S) > (242 ** remaining - 1) // 241:
            ok, policy = super()._search(S, remaining)
            return ("TRUE" if ok else "FALSE"), policy
        unknown_key = (key, width)
        if unknown_key in self.shortlist_unknown:
            self.stats["unknown_cache_hits"] += 1
            return "UNKNOWN", None
        self.stats["states"] += 1
        local_start = allowance[0]
        # Let sibling actions continue after a difficult child exhausts its
        # allowance. Each action has its own cap, within its parent's cap.
        actions = list(self._probes(S)[:width])
        if self.weights is not None and self.weights[S].sum() > 0:
            self._check_budget()
            original = int(choose(S, self.weights))
            self.stats["heuristic_calls"] += 1
            if original not in actions:
                actions.append(original)
        had_local_cutoff = False
        for g in actions:
            self._check_budget()
            if allowance[0] <= 0:
                self.stats["local_budget_unknown"] += 1
                return "UNKNOWN", None
            self._spend_guesses(1)
            codes = Q[g, S]
            if np.all(codes == codes[0]) and codes[0] != ALL_GREEN:
                self.stats["zero_progress_skipped"] += 1
                continue
            parts = [(int(code), S[codes == code]) for code in np.unique(codes)
                     if code != ALL_GREEN]
            parts.sort(key=lambda item: (-len(item[1]), item[0]))
            cap = min(allowance[0], {3: 96, 4: 256}.get(remaining, 768))
            action_budget = [cap]
            branches, complete = {}, True
            for code, part in parts:
                status, child = self._fast(part, remaining - 1, width, action_budget)
                if status != "TRUE":
                    complete = False
                    had_local_cutoff |= action_budget[0] <= 0
                    break
                branches[str(code)] = child
            allowance[0] -= cap - action_budget[0]
            if complete:
                policy = {"guess": int(g), "branches": branches}
                self.memo[key] = (True, policy)
                return "TRUE", policy
        # This cache only avoids repeating the same incomplete shortlist. It is
        # never interpreted as mathematical impossibility or stored in memo.
        if not had_local_cutoff:
            self.shortlist_unknown.add(unknown_key)
        return "UNKNOWN", None

    def certify_initial(self, first_guess, *, widths=(4, 8, 16)):
        self.started = time.monotonic()
        self.deadline = self.started + self.max_seconds
        self.stats = {"states": 0, "guesses_examined": 0, "memo_hits": 0,
                      "heuristic_calls": 0, "probe_scoring_calls": 0,
                      "probe_scoring_rows": 0, "zero_progress_skipped": 0,
                      "duplicate_partitions_skipped": 0, "local_budget_unknown": 0,
                      "unknown_cache_hits": 0}
        g = INDEX[first_guess] if isinstance(first_guess, str) else int(first_guess)
        S = np.arange(N, dtype=np.int32)
        codes = Q[g, S]
        parts = [(int(code), S[codes == code]) for code in np.unique(codes)
                 if code != ALL_GREEN]
        parts.sort(key=lambda item: (-len(item[1]), item[0]))
        branches, passes, unresolved = {}, [], []
        status, policy, reason = "UNKNOWN", None, "Bounded shortlist search incomplete"
        try:
            for width in widths:
                unresolved = []
                for code, part in parts:
                    self._check_budget()
                    if str(code) in branches:
                        continue
                    result, child = self._fast(part, 5, width, [2000 * width])
                    if result == "FALSE":
                        status, reason = "FALSE", "A first-feedback bucket is proven impossible in five turns"
                        return self._finish(status, None, reason, branches, parts, passes)
                    if result == "TRUE":
                        branches[str(code)] = child
                    else:
                        unresolved.append({"code": code, "candidates": len(part)})
                passes.append({"width": width, "certified_buckets": len(branches),
                               "unresolved": unresolved, "elapsed_seconds": time.monotonic() - self.started})
                print(f"width={width} certified={len(branches)}/{len(parts)} "
                      f"unresolved={len(unresolved)} elapsed={time.monotonic()-self.started:.1f}s", flush=True)
                if len(branches) == len(parts):
                    policy = {"guess": g, "branches": branches}
                    status, reason = "TRUE", "Complete six-turn decision tree constructed"
                    break
        except _BudgetExceeded as exc:
            reason = str(exc)
        return self._finish(status, policy, reason, branches, parts, passes)

    def _finish(self, status, policy, reason, branches, parts, passes):
        self.stats["elapsed_seconds"] = time.monotonic() - self.started
        self.stats["memo_entries"] = len(self.memo)
        return {"status": status, "policy": policy, "reason": reason,
                "stats": dict(self.stats), "passes": passes,
                "certified_first_feedback_buckets": len(branches),
                "total_first_feedback_buckets": len(parts),
                "unresolved_first_feedback_buckets": [
                    {"code": code, "candidates": len(part)}
                    for code, part in parts if str(code) not in branches]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--first", default="salet")
    parser.add_argument("--prior-day", default="2026-09-26")
    parser.add_argument("--output", default=os.path.join(RESULTS, "six_turn_salet_fast_certificate.json"))
    args = parser.parse_args()
    import backtest as bt
    guard = FastCertificateSearch(weights=bt.weights("P2s", args.prior_day), max_seconds=args.seconds)
    result = guard.certify_initial(args.first)
    verified = result["status"] == "TRUE" and verify_certificate(np.arange(N), 6, result["policy"])
    if result["status"] == "TRUE" and not verified:
        raise AssertionError("Constructed tree failed independent certificate verification")
    result.update({"independently_verified": bool(verified), "candidate_count": N,
                   "remaining": 6, "first_guess": args.first, "prior_ordering_day": args.prior_day,
                   "scope": "Every word in data/allowed.txt; normal mode, unrestricted legal probes",
                   "allowed_sha256": hashlib.sha256(open(os.path.join(DATA, "allowed.txt"), "rb").read()).hexdigest()})
    with open(args.output, "w") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "policy"}), flush=True)


if __name__ == "__main__":
    main()
