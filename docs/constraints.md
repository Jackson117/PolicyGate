# Hard constraints

Five rules fixed before any code was written. Source comments cite them by
number — "constraint #2" means the second one here. They are listed with the
reasoning because each one costs something, and the cost is the point.

### 1. Python `>=3.12,<3.14`

What the pinned benchmark checkout supports. Not 3.11.

### 2. `src/policygate/harness/` imports no agent framework

Not LangGraph, not LiteLLM, not tau2 itself — Pydantic and the standard library
only, with `deepdiff` for world diffs. `scripts/check_no_framework_deps.py`
enforces it in CI.

Two reasons. The harness is the original contribution, and handing it to a
framework would mean outsourcing the part worth showing. And a rule that is a
pure function over a proposed call and the session's facts can be unit-tested
without a model, a key or a network — which is why CI runs 276 tests with none
of them.

The cost is real: the τ²-bench binding lives in `adapters/` and the LangGraph
deployment in `service/`, so anything crossing that line is written twice.

### 3. The audit store is its own database, never τ²'s world state

τ²-bench scores a task by hashing the world-state database after the
conversation. One extra byte written there fails every task in the run. So the
audit chain gets its own store — SQLite by default, Postgres in the deployment —
and nothing in `harness/` or `store/` can reach the environment's database.

### 4. τ²-bench is pinned

Commit `a2c0247`, recorded in [`third_party/PIN.md`](../third_party/PIN.md) and
in every run's provenance. It is under active development and its internal API
has broken custom agent implementations before; an experiment whose benchmark
moves underneath it measures nothing.

### 5. Cost is a first-class constraint

A cheap model and a small subset while iterating; the full plan only for a
final report. Every run goes through a budget guard that refuses to start, or
to continue, when projected spend exceeds its line, and books actual spend per
chunk to the run's ledger. W4 came to $61.25 over 4158 simulations.
