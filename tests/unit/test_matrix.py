from __future__ import annotations

from intentbench.matrix import budget_comparison
from intentbench.schemas import ModelRole


def test_budget_comparison_uses_registered_absolute_difference_and_boundary() -> None:
    valid = budget_comparison(
        role=ModelRole.PRIMARY_DECISION,
        b2b_total_tokens=1000,
        b3_total_tokens=1100,
        maximum_allowed=0.10,
        usage_complete=True,
    )
    confounded = budget_comparison(
        role=ModelRole.WEAK_DECISION,
        b2b_total_tokens=1000,
        b3_total_tokens=1101,
        maximum_allowed=0.10,
        usage_complete=True,
    )
    incomplete = budget_comparison(
        role=ModelRole.CROSS_PROVIDER_REFERENCE,
        b2b_total_tokens=1000,
        b3_total_tokens=1000,
        maximum_allowed=0.10,
        usage_complete=False,
    )
    assert valid.status == "valid"
    assert confounded.status == "budget-confounded"
    assert incomplete.status == "usage-incomplete"
