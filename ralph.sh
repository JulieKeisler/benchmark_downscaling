#!/usr/bin/env bash
# ralph.sh — the Ralph loop, in its purest form: an agent fed the same prompt
# every iteration, with the plan (fix_plan.md) and specs re-allocated each loop.
#
# Usage:
#   AGENT_CMD="opencode run" ./ralph.sh          # default: opencode
#   AGENT_CMD="claude -p"     ./ralph.sh         # Claude Code
#   MAX_ITERS=10              ./ralph.sh         # stop after 10 loops (0 = forever)
#
# One item per loop. The agent itself runs the unit tests (backpressure) and is
# told to commit/push/tag only when they pass. See PROMPT.md.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Tool that accepts the prompt on argv (or stdin if your tool prefers).
AGENT_CMD="${AGENT_CMD:-opencode run}"
MAX_ITERS="${MAX_ITERS:-0}"        # 0 = run forever
SLEEP_BETWEEN="${SLEEP_BETWEEN:-5}" # seconds between loops

if [ ! -f PROMPT.md ]; then
  echo "PROMPT.md not found in $REPO_DIR" >&2
  exit 1
fi
if [ ! -f fix_plan.md ]; then
  echo "fix_plan.md not found in $REPO_DIR" >&2
  exit 1
fi

i=0
while :; do
  i=$((i + 1))
  echo "===== Ralph loop iteration $i ($(date -Is)) ====="

  # Feed the loop prompt. Use the command verbatim so both `opencode run` and
  # `claude -p` style CLIs work.
  $(echo "$AGENT_CMD") "$(cat PROMPT.md)" || echo "[ralph] agent exited non-zero (continuing)"

  # Guardrail reminder: the agent is expected to have run the tests and, on
  # success, committed/pushed/tagged. We do not run git here so a broken loop
  # cannot commit on the human's behalf silently.
  if [ "$MAX_ITERS" -ne 0 ] && [ "$i" -ge "$MAX_ITERS" ]; then
    echo "[ralph] reached MAX_ITERS=$MAX_ITERS — stopping."
    break
  fi
  sleep "$SLEEP_BETWEEN"
done
