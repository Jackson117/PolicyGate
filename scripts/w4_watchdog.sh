#!/bin/bash
# Run w4_run.py, restarting it when it stops making progress.
#
# W4 v3 wedged for 15 hours: every LLM request returned litellm.Timeout while the
# same endpoint answered a curl probe in two seconds, so the process -- not the
# provider -- was stuck. A dead process is visible; a live one making no progress
# is not, so progress is measured rather than assumed: tau2 rewrites a chunk's
# checkpoint as each simulation finishes, and that file's mtime is the signal.
#
# Restarting is safe and loses nothing: the runner re-claims the same directory
# (same run_id, same code identity) and tau2 resumes the chunk's checkpoint.
#
# Usage: scripts/w4_watchdog.sh <out_dir> <budget_usd> [stall_seconds] [max_restarts]
set -u

OUT_DIR=${1:?usage: w4_watchdog.sh <out_dir> <budget_usd> [stall_seconds] [max_restarts]}
BUDGET=${2:?budget required}
STALL=${3:-1500}
MAX_RESTARTS=${4:-50}

CHECKPOINTS="third_party/tau2-bench/data/simulations"
LOG="$OUT_DIR/run.log"

progress_age() {
    # Seconds since the newest checkpoint or ledger write; huge if neither exists.
    python3 - "$OUT_DIR" "$CHECKPOINTS" <<'PY'
import sys, time
from pathlib import Path
out, checkpoints = Path(sys.argv[1]), Path(sys.argv[2])
run_id = ""
plan = out / "plan.json"
if plan.exists():
    import json
    run_id = json.loads(plan.read_text()).get("run_id", "")
watched = [out / "ledger.json"]
if run_id:
    watched += list((checkpoints).glob(f"*{run_id}*/**/results.json"))
times = [p.stat().st_mtime for p in watched if p.exists()]
print(int(time.time() - max(times)) if times else 10**9)
PY
}

restarts=0
while true; do
    echo "=== watchdog: starting run at $(date -u +%FT%TZ) (restart $restarts) ===" >> "$LOG"
    uv run python scripts/w4_run.py --budget "$BUDGET" --max-concurrency 4 \
        --out "$OUT_DIR" >> "$LOG" 2>&1 &
    pid=$!

    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        age=$(progress_age)
        if [ "$age" -gt "$STALL" ]; then
            echo "=== watchdog: no progress for ${age}s, restarting at $(date -u +%FT%TZ) ===" >> "$LOG"
            pkill -9 -f "scripts/w4_run.py" 2>/dev/null
            kill -9 "$pid" 2>/dev/null
            break
        fi
    done

    wait "$pid"
    status=$?
    # 0 is a complete run; 2 is a deliberate stop (budget, infrastructure share),
    # which the runner has already reported and which restarting would not fix.
    if [ "$status" -eq 0 ] || [ "$status" -eq 2 ]; then
        echo "exit=$status" >> "$LOG"
        exit "$status"
    fi

    restarts=$((restarts + 1))
    if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
        echo "=== watchdog: giving up after $MAX_RESTARTS restarts ===" >> "$LOG"
        echo "exit=$status" >> "$LOG"
        exit "$status"
    fi
    sleep 10
done
