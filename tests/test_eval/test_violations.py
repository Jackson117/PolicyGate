"""Tests for eval/violations.py.

Built from real tau2 message and result types, because the module's only job is
to read them correctly. Two properties matter beyond the counting: a run with no
trajectory is excluded rather than scored as compliant, and a run that failed
with a trajectory is still read.
"""

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason

from policygate.eval.passk import compare_paired, rate
from policygate.eval.violations import (
    CLAUSES,
    Clause,
    count_turns,
    summarise_violations,
    violation_table,
)


def says(text: str) -> AssistantMessage:
    return AssistantMessage(role="assistant", content=text)


def calls(*names: str, text: str | None = None) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=text,
        tool_calls=[ToolCall(id=f"c{i}", name=n, arguments={}) for i, n in enumerate(names)],
    )


def run(task_id: str, trial: int, messages, *, reason=TerminationReason.USER_STOP, **kw):
    return SimulationRun(
        id=f"{task_id}-{trial}",
        task_id=task_id,
        start_time="t0",
        end_time="t1",
        duration=1.0,
        termination_reason=reason,
        messages=messages,
        trial=trial,
        **kw,
    )


USER = UserMessage(role="user", content="hi")
RESULT = ToolMessage(id="c0", role="tool", content="{}", requestor="assistant")


def test_the_clauses_quote_the_pinned_policy_verbatim():
    from tau2.utils.utils import DATA_DIR

    policy = (DATA_DIR / "tau2" / "domains" / "retail" / "policy.md").read_text()
    line_20 = policy.splitlines()[19]
    for ref in CLAUSES.values():
        assert ref.quote in line_20


def test_turns_are_counted_by_shape():
    counts = count_turns(
        run(
            "1",
            0,
            [
                says("Hi"),
                USER,
                calls("find_user_id_by_email"),
                RESULT,
                calls("get_order_details", "get_order_details", "get_order_details"),
                RESULT,
                calls("get_product_details", text="Let me look that up."),
                RESULT,
                says("Done."),
            ],
        )
    )
    assert counts.assistant_turns == 5
    assert counts.tool_call_turns == 3
    assert counts.parallel_turns == 1
    assert counts.max_calls_in_a_turn == 3
    assert counts.mixed_turns == 1
    assert counts.violated(Clause.PARALLEL_TOOL_CALLS)
    assert counts.violated(Clause.TEXT_WITH_TOOL_CALL)


def test_whitespace_is_not_text():
    counts = count_turns(run("1", 0, [calls("get_order_details", text="  \n")]))
    assert counts.mixed_turns == 0


def test_a_compliant_run_violates_nothing():
    counts = count_turns(run("1", 0, [says("Hi"), USER, calls("find_user_id_by_email"), RESULT]))
    assert not counts.violated(Clause.PARALLEL_TOOL_CALLS)
    assert not counts.violated(Clause.TEXT_WITH_TOOL_CALL)


def test_user_side_tool_calls_are_not_the_agent_s():
    user_calls = UserMessage(
        role="user",
        content=None,
        tool_calls=[
            ToolCall(id="u1", name="toggle_airplane_mode", arguments={}, requestor="user"),
            ToolCall(id="u2", name="check_status_bar", arguments={}, requestor="user"),
        ],
    )
    counts = count_turns(run("1", 0, [says("Hi"), user_calls]))
    assert counts.parallel_turns == 0


def test_a_run_with_no_trajectory_is_excluded_not_scored_compliant():
    table = violation_table(
        [
            run("1", 0, [calls("a", "b")]),
            run(
                "2",
                0,
                [],
                reason=TerminationReason.INFRASTRUCTURE_ERROR,
                info={"error_type": "APIError"},
            ),
        ],
        Clause.PARALLEL_TOOL_CALLS,
    )
    assert table.by_task() == {"1": (True,)}
    assert table.excluded == {"infrastructure_error:APIError": 1}


def test_a_failed_run_with_a_trajectory_is_still_read():
    """An agent error ends the run; what the agent did before that still happened."""
    table = violation_table(
        [run("1", 0, [calls("a", "b")], reason=TerminationReason.AGENT_ERROR)],
        Clause.PARALLEL_TOOL_CALLS,
    )
    assert table.by_task() == {"1": (True,)}
    assert table.n_excluded == 0


def test_the_table_feeds_passk_like_any_other_observation():
    sims = [
        run(str(i), t, [calls("a", "b") if i < 3 else calls("a")])
        for i in range(6)
        for t in range(2)
    ]
    table = violation_table(sims, Clause.PARALLEL_TOOL_CALLS)
    estimate = rate(table, name="parallel", resamples=500)
    assert estimate.value == 0.5
    assert estimate.n_tasks == 6


def test_an_enforcing_arm_scores_zero_by_construction():
    """The zero is the mechanism being wired, and the paired test says so loudly."""
    baseline = [run(str(i), 0, [calls("a", "b")]) for i in range(8)]
    enforced = [run(str(i), 0, [calls("a"), RESULT, calls("b")]) for i in range(8)]
    result = compare_paired(
        violation_table(enforced, Clause.PARALLEL_TOOL_CALLS),
        violation_table(baseline, Clause.PARALLEL_TOOL_CALLS),
        name="parallel_call_violation_rate",
        resamples=500,
    )
    assert result.statistic == "parallel_call_violation_rate"
    assert result.treatment == 0.0
    # Every task is higher in the control arm -- which, for violations, is worse.
    assert (result.treatment_higher, result.control_higher) == (0, 8)
    assert result.difference.value == -1.0


def test_summary_counts_violating_simulations_per_clause():
    summary = summarise_violations(
        [
            run("1", 0, [calls("a", "b")]),
            run("2", 0, [calls("a", text="checking")]),
            run("3", 0, [], reason=TerminationReason.INFRASTRUCTURE_ERROR),
        ]
    )
    assert (summary.simulations, summary.readable) == (3, 2)
    assert summary.violating == {Clause.PARALLEL_TOOL_CALLS: 1, Clause.TEXT_WITH_TOOL_CALL: 1}
