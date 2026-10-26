#!/usr/bin/env bash
# Collect what the dashboard image needs but git does not carry.
#
# Lightsail container services have no volume to mount, so the run data the
# dashboard reads has to be inside the image. That data is deliberately not in
# git -- one results chunk is megabytes and every ablation cell produces more --
# so it is staged here, into the build context, immediately before a build.
#
#   ./scripts/stage_deploy.sh [run ...]      default: w4_v4 w4_v4_ablation
#
# Everything it writes is under deploy/stage/, which is git-ignored.
set -euo pipefail

cd "$(dirname "$0")/.."

RUNS=("${@:-}")
[ -z "${RUNS[0]:-}" ] && RUNS=(w4_v4 w4_v4_ablation)

STAGE=deploy/stage
RAG_RESULTS=${POLICYGATE_RAG_RESULTS_HOST:-../policygate-rag/results}

rm -rf "$STAGE"
mkdir -p "$STAGE/runs" "$STAGE/rag"

for run in "${RUNS[@]}"; do
  if [ ! -d "experiments/$run" ]; then
    echo "no such run: experiments/$run" >&2
    exit 1
  fi
  cp -R "experiments/$run" "$STAGE/runs/$run"
  echo "staged experiments/$run ($(du -sh "$STAGE/runs/$run" | cut -f1))"
done

# The banking study's published results: small, and the only thing its page
# needs. The Retrieval page needs the sibling checkout itself and stays off in
# the container -- it says so rather than breaking.
if [ -d "$RAG_RESULTS" ]; then
  cp -R "$RAG_RESULTS/." "$STAGE/rag/"
  echo "staged $RAG_RESULTS ($(du -sh "$STAGE/rag" | cut -f1))"
else
  echo "no rag results at $RAG_RESULTS; the Banking RAG page will say so"
fi

echo "total staged: $(du -sh "$STAGE" | cut -f1)"
