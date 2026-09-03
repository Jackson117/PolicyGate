#!/usr/bin/env bash
# Restore uv's editable installs after macOS re-hides the .pth files.
#
# Background: a background process on this machine re-stamps newly created files
# with the macOS UF_HIDDEN flag, repeatedly and within minutes. CPython's
# site.addpackage() SILENTLY skips hidden .pth files, so the editable installs
# stop resolving at unpredictable moments: `import tau2` works right after
# `uv sync`, then the same command raises ModuleNotFoundError a few minutes
# later while everything in .venv still looks correct.
#
# Do NOT "fix" this by symlinking the packages into site-packages: tau2 derives
# its data directory from tau2.__file__, so a symlinked import makes it look for
# data/ under .venv and every task load fails with FileNotFoundError.
#
# Belt and braces: chflags here, plus PYTHONPATH in the Makefile (see envrc
# target) so a re-hide between this script and the actual run cannot bite.
set -euo pipefail
[[ "$(uname)" == "Darwin" ]] || exit 0
shopt -s nullglob
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
files=("$repo"/.venv/lib/python*/site-packages/*.pth)
if (( ${#files[@]} )); then
  chflags nohidden "${files[@]}"
  echo "unhid ${#files[@]} .pth file(s)"
fi
