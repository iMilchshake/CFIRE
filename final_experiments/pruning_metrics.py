from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Set, Tuple, TypedDict

import numpy as np

RuleKey = Tuple[int, int]  # (class_id, clause_id)

class PerfDict(TypedDict, total=False):
    accuracy: float  # extend later if needed (e.g., support, f1)

@dataclass(frozen=True)
class RuleMetrics:
    # structure
    clause_keys: List[RuleKey]                     # stable order of (class, clause)
    match_per_sample: List[List[RuleKey]]          # for each sample: matched rule keys

    # rule quality / winners
    perf_by_key: Dict[RuleKey, PerfDict]           # (class,clause) -> {'accuracy': ...}
    wins: np.ndarray                               # shape [n_clauses]
    loss: np.ndarray                               # shape [n_clauses]
    winner_key_per_sample: List[Optional[RuleKey]] # length = n_samples

    # extra descriptive stats (no printing)
    coverage_per_rule: np.ndarray                  # how often each rule matched
    match_hist: Dict[int, int]                     # #matches -> count of samples
    share_multi: float                             # % of samples with ≥2 matches
    collision_ratio: Dict[str, float]              # {'intra': %, 'inter': %}

class CFIRELike(Protocol):
    @property
    def dnf(self) -> object: ...
    def __call__(self, X, explain: bool = False):
        """When explain=True, should return either:
           - iterable of (pred, matches) with matches: List[Tuple[RuleKey, any_payload]]
           - OR directly List[List[RuleKey]]
        """

def canon(rule: Sequence) -> Tuple:
    """Canonicalize rule so it matches keys in rule_performances."""
    return tuple(rule) if isinstance(rule, list) else rule

def _normalize_matches(explain_out) -> List[List[RuleKey]]:
    """Normalize CFIRE explain output to List[List[RuleKey]].

    Supported:
      A) (preds, matches_per_sample)
      B) iterable of (pred, matches) per sample
      C) already List[List[RuleKey]]
    Where each `matches` can be:
      - None  -> treated as []
      - List[Tuple[RuleKey, any_payload]]
      - List[RuleKey]
    """
    # Case A: top-level tuple
    if isinstance(explain_out, tuple) and len(explain_out) == 2:
        _, matches_global = explain_out
        return _normalize_matches(matches_global)

    # Ensure indexable sequence
    try:
        first = explain_out[0]  # type: ignore[index]
    except Exception:
        explain_out = list(explain_out)
        first = explain_out[0] if explain_out else None

    # Case B: list of (pred, matches) pairs per-sample
    if isinstance(first, tuple) and len(first) == 2:
        out: List[List[RuleKey]] = []
        for _, matches in explain_out:  # type: ignore[assignment]
            if matches is None:
                out.append([])
                continue
            # matches might be List[(RuleKey, payload)] or List[RuleKey]
            if isinstance(matches, list) and matches:
                m0 = matches[0]
                # List[(RuleKey, payload)]
                if isinstance(m0, tuple) and len(m0) >= 2 and isinstance(m0[0], tuple):
                    out.append([k for (k, _) in matches])
                # List[RuleKey]
                elif isinstance(m0, tuple) and len(m0) == 2 and all(isinstance(x, (int, np.integer)) for x in m0):
                    out.append(matches)
                else:
                    out.append([])
            else:
                out.append([])
        return out

    # Case C: already List[List[RuleKey]]
    return explain_out

def _build_perf_by_key(cf: CFIRELike) -> Dict[RuleKey, PerfDict]:
    """Extract per‑rule accuracy from cf.dnf.rule_performances mapping."""
    out: Dict[RuleKey, PerfDict] = {}
    rules = getattr(cf.dnf, "rules")
    perf = getattr(cf.dnf, "rule_performances")
    for cls_id, class_rules in enumerate(rules):
        for cid, rule in enumerate(class_rules):
            pd = perf[cls_id][canon(rule)]
            out[(cls_id, cid)] = {"accuracy": float(pd.get("accuracy", 0.0))}
    return out

def _compute_winners(
        match_per_sample: List[List[RuleKey]],
        clause_keys: List[RuleKey],
        perf_by_key: Dict[RuleKey, PerfDict],
) -> tuple[np.ndarray, np.ndarray, List[Optional[RuleKey]]]:
    """Winner = matched rule with max per‑rule accuracy."""
    key2col = {k: i for i, k in enumerate(clause_keys)}

    def best_key(keys: List[RuleKey]) -> RuleKey:
        return max(keys, key=lambda k: float(perf_by_key.get(k, {}).get("accuracy", 0.0)))

    n_samples, n_clauses = len(match_per_sample), len(clause_keys)
    winner_idx = np.full(n_samples, -1, dtype=int)
    wins = np.zeros(n_clauses, dtype=int)
    loss = np.zeros(n_clauses, dtype=int)

    for s, keys in enumerate(match_per_sample):
        if not keys:
            continue
        w = best_key(keys)
        w_col = key2col[w]
        winner_idx[s] = w_col
        for k in keys:
            (wins if k == w else loss)[key2col[k]] += 1

    winner_key_per_sample: List[Optional[RuleKey]] = [
        clause_keys[idx] if idx != -1 else None
        for idx in winner_idx
    ]
    return wins, loss, winner_key_per_sample

def _coverage_hist_collision(
        clause_keys: List[RuleKey],
        match_per_sample: List[List[RuleKey]],
        wins: np.ndarray,
) -> tuple[np.ndarray, Dict[int, int], float, Dict[str, float]]:
    """
    coverage_per_rule, match_hist, share_multi, collision_ratio
    """
    n_samples, n_clauses = len(match_per_sample), len(clause_keys)
    key2col = {k: i for i, k in enumerate(clause_keys)}

    M = np.zeros((n_samples, n_clauses), dtype=bool)
    coverage = np.zeros(n_clauses, dtype=int)
    match_counts = np.zeros(n_samples, dtype=int)

    for s, keys in enumerate(match_per_sample):
        match_counts[s] = len(keys)
        for k in keys:
            j = key2col[k]
            M[s, j] = True
            coverage[j] += 1

    # histogram of #matches per sample
    uniq, cnts = np.unique(match_counts, return_counts=True)
    match_hist = {int(k): int(v) for k, v in zip(uniq.tolist(), cnts.tolist())}
    share_multi = float((match_counts >= 2).mean() * 100.0)

    # collision ratio based on winner's class
    # (we don't need 'wins' here strictly; included for symmetry with original)
    intra_res = 0
    inter_res = 0
    # For collision we need the winner per sample; recompute quickly with "most frequent winner class":
    # to avoid recomputing, we’ll compute using majority class among matched vs any other class present.
    #  here we just return placeholders; real computation happens in compute_rule_metrics with true winners.
    # We'll just return zeros here; compute_rule_metrics will overwrite with correct ratio.
    collision_ratio = {"intra": 0.0, "inter": 0.0}

    return coverage, match_hist, share_multi, collision_ratio

def _collision_ratio_from_winners(
        match_per_sample: List[List[RuleKey]],
        winner_key_per_sample: List[Optional[RuleKey]],
) -> Dict[str, float]:
    intra_res = 0
    inter_res = 0
    for keys, w in zip(match_per_sample, winner_key_per_sample):
        if len(keys) <= 1 or w is None:
            continue
        other_cls = {k[0] for k in keys} - {w[0]}
        if other_cls:
            inter_res += 1
        else:
            intra_res += 1
    total = intra_res + inter_res
    if total == 0:
        return {"intra": 0.0, "inter": 0.0}
    return {
        "intra": intra_res / total * 100.0,
        "inter": inter_res / total * 100.0,
    }


def compute_rule_metrics(cfire: CFIRELike, X_val) -> RuleMetrics:
    """
    Single call that the experiment orchestrator can use.
    Returns a RuleMetrics object with everything needed for pruning decisions.
    """
    # matches
    explain_out = cfire(X_val, explain=True)
    match_per_sample = _normalize_matches(explain_out)

    # stable key list by first appearance
    clause_keys: List[RuleKey] = []
    seen: Set[RuleKey] = set()
    for keys in match_per_sample:
        for k in keys:
            if k not in seen:
                seen.add(k)
                clause_keys.append(k)

    # per-rule perf and winners
    perf_by_key = _build_perf_by_key(cfire)
    wins, loss, winner_key_per_sample = _compute_winners(match_per_sample, clause_keys, perf_by_key)

    # descriptive stats
    coverage_per_rule, match_hist, share_multi, _ = _coverage_hist_collision(
        clause_keys, match_per_sample, wins
    )
    collision_ratio = _collision_ratio_from_winners(match_per_sample, winner_key_per_sample)

    return RuleMetrics(
        clause_keys=clause_keys,
        match_per_sample=match_per_sample,
        perf_by_key=perf_by_key,
        wins=wins,
        loss=loss,
        winner_key_per_sample=winner_key_per_sample,
        coverage_per_rule=coverage_per_rule,
        match_hist=match_hist,
        share_multi=share_multi,
        collision_ratio=collision_ratio,
    )
