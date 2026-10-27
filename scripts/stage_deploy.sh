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

# The banking study's published results: small, and all its page needs.
if [ -d "$RAG_RESULTS" ]; then
  cp -R "$RAG_RESULTS/." "$STAGE/rag/"
  echo "staged $RAG_RESULTS ($(du -sh "$STAGE/rag" | cut -f1))"
else
  echo "no rag results at $RAG_RESULTS; the Banking RAG page will say so"
fi

# The Retrieval page runs the study's own retrieval, and that code resolves the
# corpus and the embedding cache relative to its repository root -- three
# parents up from the module. So it needs the checkout's shape, not a wheel:
# src/ on PYTHONPATH with index_cache/, runs/ledger.json and the banking
# documents beside it. Without the sibling the page says it is unavailable,
# which is why the directory is created either way.
RAG_REPO=${POLICYGATE_RAG_REPO:-$(dirname "$RAG_RESULTS")}
PKG="$STAGE/rag-pkg"
mkdir -p "$PKG"
touch "$PKG/.keep"   # an empty directory is not a COPY source
DOCS=third_party/tau2-bench/data/tau2/domains/banking_knowledge/documents
if [ -d "$RAG_REPO/src/policygate_rag" ] && [ -d "$RAG_REPO/index_cache" ]; then
  mkdir -p "$PKG/src" "$PKG/runs" "$PKG/$(dirname "$DOCS")"
  cp -R "$RAG_REPO/src/policygate_rag" "$PKG/src/"
  cp -R "$RAG_REPO/index_cache" "$PKG/"
  [ -f "$RAG_REPO/runs/ledger.json" ] && cp "$RAG_REPO/runs/ledger.json" "$PKG/runs/"
  [ -d "$RAG_REPO/$DOCS" ] && cp -R "$RAG_REPO/$DOCS" "$PKG/$(dirname "$DOCS")/"
  echo "staged the retrieval inspector ($(du -sh "$PKG" | cut -f1))"
else
  echo "no policygate-rag checkout at $RAG_REPO; the Retrieval page will say so"
fi

echo "total staged: $(du -sh "$STAGE" | cut -f1)"
