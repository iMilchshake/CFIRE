from __future__ import annotations

from dataclasses import dataclass
from typing import  Sequence, Set

from final_experiments.metrics import Rules
from final_experiments.pruning_metrics import RuleKey, RuleMetrics


@dataclass(frozen=True)
class PruningDecision:
    to_remove: Set[RuleKey]
    rationale: str

def decide_by_wins(metrics: RuleMetrics, win_threshold: int) -> PruningDecision:
    """
    Classic policy from the original script:
    remove rules whose absolute win count <= win_threshold.
    """
    to_remove: Set[RuleKey] = {
        metrics.clause_keys[i]
        for i, w in enumerate(metrics.wins)
        if int(w) <= int(win_threshold)
    }
    rationale = f"Removed {len(to_remove)}/{len(metrics.clause_keys)} rules with wins <= {win_threshold}"
    return PruningDecision(to_remove=to_remove, rationale=rationale)

def prune_rules(rule_tree: Sequence[Sequence], to_remove: Set[RuleKey]) -> Rules:
    """
    Pure function: return a new rules structure with (class, clause) keys filtered out.
    No side effects; you decide outside whether to set cfire.dnf.rules = new_rules.
    """
    return [
        [r for cid, r in enumerate(rules) if (cls, cid) not in to_remove]
        for cls, rules in enumerate(rule_tree)
    ]
