#!/bin/bash
# SessionStart hook: install the package + test deps so tests run in
# Claude Code on the web sessions. Synchronous, idempotent, web-only.
set -euo pipefail

# Only run in remote (Claude Code on the web) sessions.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

# Light, test-ready env: requests + numpy (core) + pytest (dev extra).
# Deliberately NOT installing ".[ml]" (torch etc.) — the offline test suite
# doesn't need it, and it would make every session start slow.
pip install --quiet --disable-pip-version-check -e ".[dev]" 1>&2

echo "peaks: installed (core + dev). Run 'pytest' for the offline suite." 1>&2
