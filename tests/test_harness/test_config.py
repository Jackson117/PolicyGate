"""Contract tests for harness/config.py.

The ablation matrix is generated from HarnessConfig, so these tests guard the
property CLAUDE.md cares about: a mechanism that exists has a switch, and a
switch that exists produces a row.
"""

import pytest
from pydantic import ValidationError

from policygate.harness.config import (
    ApprovalConfig,
    ApprovalPolicy,
    HarnessConfig,
    LoopGuardConfig,
)


class TestArms:
    def test_baseline_enables_nothing(self):
        assert HarnessConfig.baseline().enabled_ablation_switches() == ()

    def test_full_only_claims_mechanisms_that_exist(self):
        """A recorded arm must not name a switch with no implementation behind it.

        Idempotency is deferred past W3 (W1 saw zero duplicate write calls); the
        other three have switches but no module yet. Move a name out of this set
        in the same commit as the module that implements it.
        """
        full = HarnessConfig.full()
        off = set(HarnessConfig.ablation_switches()) - set(full.enabled_ablation_switches())
        assert off == {
            "idempotency_enabled",
            "classified_retry_enabled",
            "compensation_enabled",
            "loop_guards_enabled",
        }

    def test_arms_are_immutable(self):
        with pytest.raises(ValidationError):
            HarnessConfig.baseline().gating_enabled = True

    def test_full_accepts_overrides(self):
        assert HarnessConfig.full(idempotency_enabled=True).idempotency_enabled is True

    def test_full_can_override_a_switch_it_sets_itself(self):
        # The earlier version splatted **overrides next to explicit keywords, so
        # this raised TypeError -- and it is the exact call an ablation arm makes.
        assert HarnessConfig.full(gating_enabled=False).gating_enabled is False
        assert HarnessConfig.full(gating_enabled=False).schema_check_enabled is True


class TestAblationSwitches:
    def test_switches_are_discovered_from_the_model_not_hand_listed(self):
        # Every boolean field is either ablatable or explicitly excluded; this is
        # what stops a new mechanism from silently missing its ablation row.
        booleans = {n for n, f in HarnessConfig.model_fields.items() if f.annotation is bool}
        assert booleans == set(HarnessConfig.ablation_switches()) | set(HarnessConfig.NON_ABLATABLE)

    def test_audit_is_excluded_because_it_cannot_move_the_reward(self):
        assert "audit_enabled" not in HarnessConfig.ablation_switches()
        assert HarnessConfig.full().audit_enabled is True

    def test_each_core_contribution_has_a_switch(self):
        switches = set(HarnessConfig.ablation_switches())
        assert {"gating_enabled", "idempotency_enabled"} <= switches

    def test_the_approval_gate_is_out_of_the_matrix_and_off_everywhere(self):
        """Under a fixed auto_approve it cannot change a trajectory; its row would be flat."""
        assert "approval_gate_enabled" not in HarnessConfig.ablation_switches()
        assert HarnessConfig.full().approval_gate_enabled is False
        assert HarnessConfig.full().approval.policy is ApprovalPolicy.AUTO_APPROVE

    def test_validation_mechanisms_are_separately_ablatable(self):
        # W1 showed these are distinct failure modes with distinct evidence
        # (task 22, task 14, "Order not found"); one shared flag could not
        # attribute an improvement to any of them.
        switches = set(HarnessConfig.ablation_switches())
        assert {
            "value_canonicalization_enabled",
            "referent_check_enabled",
            "precondition_check_enabled",
            "parallel_call_block_enabled",
        } <= switches


class TestOneAtATime:
    def test_only_enables_exactly_the_named_switch(self):
        for switch in HarnessConfig.full().enabled_ablation_switches():
            arm = HarnessConfig.only(switch)
            assert arm.enabled_ablation_switches() == (switch,)
            assert arm.name == f"only_{switch}"

    def test_only_refuses_what_is_not_ablatable(self):
        with pytest.raises(ValueError, match="not an ablatable switch"):
            HarnessConfig.only("approval_gate_enabled")
        with pytest.raises(ValueError, match="not an ablatable switch"):
            HarnessConfig.only("no_such_switch")

    def test_only_accepts_non_behavioural_overrides(self):
        arm = HarnessConfig.only("gating_enabled", audit_enabled=True)
        assert arm.audit_enabled is True
        assert arm.enabled_ablation_switches() == ("gating_enabled",)


class TestLeaveOneOut:
    def test_disables_only_the_named_switch(self):
        full = HarnessConfig.full()
        arm = full.without("gating_enabled")
        assert arm.gating_enabled is False
        assert arm.value_canonicalization_enabled is True

    def test_names_the_arm_after_what_it_removed(self):
        assert "no_gating_enabled" in HarnessConfig.full().without("gating_enabled").name

    def test_can_remove_several_at_once(self):
        arm = HarnessConfig.full().without("gating_enabled", "context_pruning_enabled")
        assert (arm.gating_enabled, arm.context_pruning_enabled) == (False, False)

    def test_rejects_an_unknown_switch(self):
        with pytest.raises(ValueError, match="not ablatable"):
            HarnessConfig.full().without("no_such_switch")

    def test_rejects_a_non_ablatable_switch(self):
        with pytest.raises(ValueError, match="not ablatable"):
            HarnessConfig.full().without("audit_enabled")


class TestApprovalConfig:
    def test_noise_rate_is_rejected_outside_the_noisy_oracle(self):
        with pytest.raises(ValidationError):
            ApprovalConfig(policy=ApprovalPolicy.ORACLE, noise_rate=0.2)

    def test_noisy_oracle_accepts_a_noise_rate(self):
        assert ApprovalConfig(policy=ApprovalPolicy.NOISY_ORACLE, noise_rate=0.2).noise_rate == 0.2

    def test_noise_rate_is_bounded(self):
        with pytest.raises(ValidationError):
            ApprovalConfig(policy=ApprovalPolicy.NOISY_ORACLE, noise_rate=1.5)

    def test_writes_are_what_pause_by_default(self):
        assert [k.value for k in ApprovalConfig().require_for_kinds] == ["write"]


class TestLoopGuards:
    def test_budgets_must_be_positive(self):
        with pytest.raises(ValidationError):
            LoopGuardConfig(max_tool_calls=0)

    def test_defaults_sit_clear_of_observed_behaviour(self):
        # Longest W1 run was 13 tool calls; a guard that trips on normal work
        # would be scored as a harness effect.
        assert LoopGuardConfig().max_tool_calls > 13

