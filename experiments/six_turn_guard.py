"""Budgeted, exact-support Wordle safety certificates.

The certificate covers EVERY answer index in S, regardless of its probability.
TRUE means a complete decision tree was constructed; FALSE means all legal
guesses were ruled out (or a necessary bound failed); a spent budget is UNKNOWN.
No true answer or history is accepted by this module.

Typical use, immediately before guess four::

    guard = SixTurnGuard(weights=w, max_seconds=10)
    result = guard.certify(S, remaining=3, preferred_guesses=[original_guess])
    if result.status == "TRUE":
        assert verify_certificate(S, 3, result.policy)
        guess = result.policy["guess"]

After observing a non-green feedback code, follow policy["branches"][str(code)].
Do not resume an unchecked heuristic: the guarantee belongs to the whole tree.
To check a proposed move, use certify_action(S, remaining, guess). UNKNOWN is
never a certificate of safety, nor is it proof that a state is impossible.

A guard first enabled on turn four cannot guarantee that earlier moves always
leave a solvable state. Full-game guarantees require an initial certificate or
certified choices at every earlier state, with a previously certified fallback.
"""

from dataclasses import dataclass
import time

import numpy as np

from common import ALL_GREEN, N, Q, choose


TRUE, FALSE, UNKNOWN = "TRUE", "FALSE", "UNKNOWN"


@dataclass
class GuardResult:
    status: str
    policy: dict | None
    reason: str
    stats: dict


class _BudgetExceeded(Exception):
    pass


def _state(S):
    values = np.asarray(S)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("S must be a one-dimensional integer array of answer indices")
    if not len(values) or np.any(values < 0) or np.any(values >= N):
        raise ValueError("S must contain at least one valid answer index")
    if len(np.unique(values)) != len(values):
        raise ValueError("S must not contain duplicate answer indices")
    return np.sort(values.astype(np.int32, copy=False))


def _remaining(value):
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError("remaining must be a positive integer")
    return int(value)


def _guess(value):
    if isinstance(value, bool) or int(value) != value or not 0 <= value < N:
        raise ValueError("guess must be a valid integer word index")
    return int(value)


def verify_certificate(S, remaining, policy):
    """Independently check complete branch coverage and the worst-case depth.

    Returns False for a malformed/incomplete certificate. The candidate set is
    supplied by the caller, so a certificate cannot silently narrow its support.
    Recomputes every feedback branch from Q, without using the search or weights.
    """
    S, remaining = _state(S), _remaining(remaining)

    def check(part, turns, node):
        if turns < 1 or not isinstance(node, dict):
            return False
        try:
            g = _guess(node["guess"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        branches = node.get("branches")
        if not isinstance(branches, dict):
            return False
        codes = Q[g, part]
        required = {str(int(c)) for c in np.unique(codes) if c != ALL_GREEN}
        if set(branches) != required:
            return False
        return all(check(part[codes == int(c)], turns - 1, branches[c]) for c in required)

    return bool(check(S, remaining, policy))


class SixTurnGuard:
    """Certify a finite-horizon tree with a fresh resource budget per call.

    Weights affect search order only. All members of S remain possible, including
    zero-weight members should a caller explicitly include them. Memoized TRUE
    and FALSE states are independent of weights and can safely survive date
    changes. UNKNOWN results and partially built trees are never memoized.

    max_nodes counts uncached states; max_guesses counts feedback rows examined
    by feasibility search (heuristic calls are counted separately).
    None disables an individual limit. The wall-clock limit is cooperative,
    checked between small batches/branches, rather than a hard process timeout.
    """

    def __init__(self, *, weights=None, max_nodes=50_000, max_guesses=250_000,
                 max_seconds=10.0):
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != (N,) or not np.isfinite(weights).all() or np.any(weights < 0):
                raise ValueError("weights must be N finite, nonnegative values")
        for name, value in (("max_nodes", max_nodes), ("max_guesses", max_guesses)):
            if value is not None and (isinstance(value, bool) or int(value) != value or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds < 0):
            raise ValueError("max_seconds must be nonnegative and finite, or None")
        self.weights = weights
        self.max_nodes = max_nodes
        self.max_guesses = max_guesses
        self.max_seconds = max_seconds
        self.memo = {}
        self.probe_order_cache = {}

    def certify(self, S, remaining, *, preferred_guesses=()):
        """Find any complete tree, trying preferred root actions first."""
        return self._run(S, remaining, tuple(_guess(g) for g in preferred_guesses), None)

    def certify_action(self, S, remaining, guess):
        """Certify a tree whose first move is exactly guess."""
        return self._run(S, remaining, (), _guess(guess))

    def _run(self, S, remaining, preferred, forced):
        S, remaining = _state(S), _remaining(remaining)
        self.started = time.monotonic()
        self.deadline = None if self.max_seconds is None else self.started + self.max_seconds
        self.stats = {"states": 0, "guesses_examined": 0, "memo_hits": 0, "heuristic_calls": 0,
                      "probe_scoring_calls": 0, "probe_scoring_rows": 0,
                      "zero_progress_skipped": 0, "duplicate_partitions_skipped": 0}
        try:
            if forced is None:
                ok, policy = self._search(S, remaining, preferred)
            else:
                self._check_budget()
                self._spend_guesses(1)
                ok, policy = self._try_guess(S, remaining, forced)
            status = TRUE if ok else FALSE
            reason = ("Complete decision tree covers every supplied candidate" if ok else
                      "All legal continuations ruled out by exhaustive search or necessary bounds")
        except _BudgetExceeded as exc:
            status, policy, reason = UNKNOWN, None, str(exc)
        self.stats["elapsed_seconds"] = time.monotonic() - self.started
        self.stats["memo_entries"] = len(self.memo)
        return GuardResult(status, policy, reason, dict(self.stats))

    def _check_budget(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise _BudgetExceeded("Wall-clock budget exhausted; feasibility is unknown")

    def _spend_guesses(self, count):
        self._check_budget()
        if self.max_guesses is not None and self.stats["guesses_examined"] + count > self.max_guesses:
            raise _BudgetExceeded("Guess-evaluation budget exhausted; feasibility is unknown")
        self.stats["guesses_examined"] += count

    def _ordered_answers(self, S):
        if self.weights is None:
            return S
        return S[np.argsort(-self.weights[S], kind="stable")]

    def _balanced_probes(self, S):
        """Rank all legal moves by worst bucket, then collision count.

        This is search ordering, not a feasibility approximation. Green is a
        completed game, so only non-green bucket sizes contribute to the rank.
        Time checks between 256-row batches keep this cooperative with budgets.
        """
        key = S.tobytes()
        if key in self.probe_order_cache:
            return self.probe_order_cache[key]
        largest = np.empty(N, dtype=np.int32)
        collisions = np.empty(N, dtype=np.int64)
        self.stats["probe_scoring_calls"] += 1
        for start in range(0, N, 256):
            self._check_budget()
            stop = min(start + 256, N)
            rows = stop - start
            bins = Q[start:stop][:, S].astype(np.int64)
            bins += (np.arange(rows) * 243)[:, None]
            counts = np.bincount(bins.ravel(), minlength=rows * 243).reshape(rows, 243)
            counts[:, ALL_GREEN] = 0
            largest[start:stop] = counts.max(axis=1)
            collisions[start:stop] = (counts * counts).sum(axis=1)
            self.stats["probe_scoring_rows"] += rows
        self._check_budget()
        order = np.lexsort((np.arange(N), collisions, largest))[:8]
        self.probe_order_cache[key] = order
        return order

    def _candidate_order(self, S, preferred=(), use_heuristic=False):
        # The shortlist changes order only. The rest of all N legal guesses is
        # still visited before FALSE may be returned.
        seen = set()
        for item in preferred:
            g = int(item)
            if g not in seen:
                seen.add(g)
                yield g
        if use_heuristic:
            for item in self._balanced_probes(S):
                g = int(item)
                if g not in seen:
                    seen.add(g)
                    yield g
        if use_heuristic and self.weights is not None and self.weights[S].sum() > 0:
            self._check_budget()
            g = int(choose(S, self.weights))
            self.stats["heuristic_calls"] += 1
            self._check_budget()
            if g not in seen:
                seen.add(g)
                yield g
        for collection in (self._ordered_answers(S), range(N)):
            for item in collection:
                g = int(item)
                if g not in seen:
                    seen.add(g)
                    yield g

    def _sequential_certificate(self, S):
        # At most |S| guesses: each unsuccessful candidate guess removes itself.
        g = int(self._ordered_answers(S)[0])
        codes = Q[g, S]
        return {"guess": g, "branches": {
            str(int(c)): self._sequential_certificate(S[codes == c])
            for c in np.unique(codes) if c != ALL_GREEN
        }}

    def _search(self, S, remaining, preferred=()):
        key = (S.tobytes(), remaining)
        if key in self.memo:
            self.stats["memo_hits"] += 1
            return self.memo[key]
        self._check_budget()
        if self.max_nodes is not None and self.stats["states"] >= self.max_nodes:
            raise _BudgetExceeded("State budget exhausted; feasibility is unknown")
        self.stats["states"] += 1

        if len(S) <= remaining:
            result = (True, self._sequential_certificate(S))
        elif remaining == 1:
            result = (False, None)
        elif len(S) > (242 ** remaining - 1) // 241:
            # One green solution plus at most 242 continuing branches per node.
            # This deliberately loose capacity is a valid necessary bound.
            result = (False, None)
        elif remaining == 2:
            result = self._two_turns(S, preferred)
        else:
            result = (False, None)
            seen_rows = set()
            for g in self._candidate_order(S, preferred, use_heuristic=True):
                self._spend_guesses(1)
                row = Q[g, S]
                signature = row.tobytes()
                if signature in seen_rows:
                    self.stats["duplicate_partitions_skipped"] += 1
                    continue
                seen_rows.add(signature)
                if np.all(row == row[0]) and row[0] != ALL_GREEN:
                    self.stats["zero_progress_skipped"] += 1
                    continue
                ok, policy = self._try_guess(S, remaining, g)
                if ok:
                    result = (True, policy)
                    break
        self.memo[key] = result
        return result

    def _two_turns(self, S, preferred):
        # In two turns, a successful first probe must give every answer its own
        # feedback pattern. A green pattern also occurs at most once, so the
        # condition is exactly injectivity of the whole feedback row over S.
        order = iter(self._candidate_order(S, preferred))
        while True:
            self._check_budget()
            batch_size = 128
            if self.max_guesses is not None:
                left = self.max_guesses - self.stats["guesses_examined"]
                if left <= 0:
                    # There might be no untested guesses; check that before
                    # turning completed exhaustive search into UNKNOWN.
                    try:
                        next(order)
                    except StopIteration:
                        return False, None
                    raise _BudgetExceeded("Guess-evaluation budget exhausted; feasibility is unknown")
                batch_size = min(batch_size, left)
            batch = []
            for _ in range(batch_size):
                try:
                    batch.append(next(order))
                except StopIteration:
                    break
            if not batch:
                return False, None
            self._spend_guesses(len(batch))
            rows = Q[np.asarray(batch)[:, None], S[None, :]]
            ordered = np.sort(rows, axis=1)
            unique = np.all(ordered[:, 1:] != ordered[:, :-1], axis=1)
            valid = np.flatnonzero(unique)
            if len(valid):
                g = batch[int(valid[0])]
                codes = Q[g, S]
                branches = {
                    str(int(code)): {"guess": int(a), "branches": {}}
                    for code, a in zip(codes, S) if code != ALL_GREEN
                }
                return True, {"guess": g, "branches": branches}

    def _try_guess(self, S, remaining, g):
        codes = Q[g, S]
        parts = [(int(c), S[codes == c]) for c in np.unique(codes) if c != ALL_GREEN]
        if not parts:
            return True, {"guess": g, "branches": {}}
        if remaining <= 1:
            return False, None
        # A forced zero-progress guess is still certifiable if enough turns
        # remain. It is skipped only in unrestricted search, where it is never
        # necessary for existence of a tree.
        parts.sort(key=lambda item: (-len(item[1]), item[0]))
        branches = {}
        for code, part in parts:
            self._check_budget()
            ok, child = self._search(part, remaining - 1)
            if not ok:
                return False, None
            branches[str(code)] = child
        return True, {"guess": g, "branches": branches}


def certify(S, remaining, *, preferred_guesses=(), weights=None,
            max_nodes=50_000, max_guesses=250_000, max_seconds=10.0):
    """Convenience wrapper; reuse SixTurnGuard to retain exact-state memoization."""
    guard = SixTurnGuard(weights=weights, max_nodes=max_nodes,
                         max_guesses=max_guesses, max_seconds=max_seconds)
    return guard.certify(S, remaining, preferred_guesses=preferred_guesses)
