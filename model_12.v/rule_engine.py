"""
rule_engine.py
--------------
Lightweight rule-based layer that runs BEFORE and AFTER the ML model.

Rules can:
  • Hard-BLOCK a transaction (bypass model entirely)
  • Hard-ALLOW a transaction
  • Escalate risk score
  • Add warning flags

Each rule returns a RuleResult namedtuple.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from config import RULE_EXTREME_AMOUNT


# ── Result types ───────────────────────────────────────────────────────────

@dataclass
class RuleResult:
    triggered:    bool   = False
    action:       str    = "none"          # "block" | "allow" | "escalate" | "none"
    rule_name:    str    = ""
    reason:       str    = ""
    score_boost:  float  = 0.0            # added to model score if action=escalate


@dataclass
class RuleEngineOutput:
    hard_decision:  Optional[str]  = None  # "block" | "allow" | None (let model decide)
    score_boost:    float          = 0.0
    triggered_rules: List[str]     = field(default_factory=list)
    warnings:       List[str]      = field(default_factory=list)


# ── Individual Rules ───────────────────────────────────────────────────────

def rule_extreme_amount(txn: dict) -> RuleResult:
    amount = txn.get("amount", 0.0)
    if amount > RULE_EXTREME_AMOUNT:
        return RuleResult(
            triggered=True, action="escalate",
            rule_name="EXTREME_AMOUNT",
            reason=f"Amount ${amount:.2f} exceeds extreme threshold ${RULE_EXTREME_AMOUNT}",
            score_boost=0.25,
        )
    return RuleResult()


def rule_missing_features(txn: dict) -> RuleResult:
    features = txn.get("features", {})
    if not features or len(features) < 10:
        return RuleResult(
            triggered=True, action="escalate",
            rule_name="MISSING_FEATURES",
            reason=f"Only {len(features)} features present — expected 28",
            score_boost=0.15,
        )
    return RuleResult()


def rule_negative_amount(txn: dict) -> RuleResult:
    amount = txn.get("amount", 0.0)
    if amount < 0:
        return RuleResult(
            triggered=True, action="block",
            rule_name="NEGATIVE_AMOUNT",
            reason=f"Negative transaction amount: ${amount:.2f}",
        )
    return RuleResult()


def rule_zero_amount(txn: dict) -> RuleResult:
    amount = txn.get("amount", 0.0)
    if amount == 0.0:
        return RuleResult(
            triggered=True, action="escalate",
            rule_name="ZERO_AMOUNT",
            reason="Zero-value transaction — may indicate card testing",
            score_boost=0.10,
        )
    return RuleResult()


def rule_micro_amount(txn: dict) -> RuleResult:
    """Very small amounts are sometimes used in card testing."""
    amount = txn.get("amount", 0.0)
    if 0 < amount < 0.50:
        return RuleResult(
            triggered=True, action="escalate",
            rule_name="MICRO_AMOUNT",
            reason=f"Micro transaction ${amount:.2f} — possible card test",
            score_boost=0.05,
        )
    return RuleResult()


# ── Master Rule Runner ─────────────────────────────────────────────────────

_PRE_MODEL_RULES = [
    rule_negative_amount,
    rule_missing_features,
    rule_extreme_amount,
    rule_zero_amount,
    rule_micro_amount,
]


def run_pre_model_rules(txn: dict) -> RuleEngineOutput:
    """
    Run all pre-model rules.
    Returns RuleEngineOutput — if hard_decision is set, skip the model.
    """
    output = RuleEngineOutput()

    for rule_fn in _PRE_MODEL_RULES:
        result = rule_fn(txn)
        if not result.triggered:
            continue

        output.triggered_rules.append(result.rule_name)
        output.warnings.append(result.reason)
        output.score_boost += result.score_boost

        if result.action == "block":
            output.hard_decision = "block"
            break                               # hard block — stop processing
        if result.action == "allow":
            output.hard_decision = "allow"
            break

    return output


def run_post_model_rules(txn: dict, model_score: float,
                         rule_output: RuleEngineOutput) -> float:
    """
    Apply score boosts from pre-model rules to model score.
    Clamp result to [0, 1].
    """
    adjusted = min(1.0, model_score + rule_output.score_boost)
    return round(adjusted, 4)
