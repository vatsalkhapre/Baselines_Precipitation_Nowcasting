#!/usr/bin/env bash
# claude_autoresume.sh
#
# Runs Claude Code, watches for rate-limit / usage-limit errors, sleeps until
# reset, and relaunches with --continue — repeating until the task itself
# prints the completion marker. Meant to run OUTSIDE Claude Code (cron,
# systemd, or just `nohup ./claude_autoresume.sh &`), since nothing inside
# Claude Code can restart Claude Code.
#
# USAGE:
#   chmod +x claude_autoresume.sh
#   nohup ./claude_autoresume.sh > /dev/null 2>&1 &
#
# Or as a systemd service (recommended — survives reboots, auto-restarts on
# crash of the wrapper itself). See notes at the bottom of this file.

set -uo pipefail

# ---------------- Config — edit these for your setup ----------------
WORKDIR="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/"
PROMPT_FILE="$WORKDIR/PROMPT/dawncast_pixel_prompt.md"
LOGDIR="$WORKDIR/claude_autoresume_logs"
LOCKFILE="/tmp/claude_autoresume.lock"

# 5h rolling window + safety margin. If you also hit the weekly cap, this
# script will keep retrying every DEFAULT_SLEEP_SECS regardless — harmless,
# it'll just keep hitting the limit message and re-sleeping until the weekly
# window actually resets too.
DEFAULT_SLEEP_SECS=$((5*60*60 + 10*60))   # 5h10m
CRASH_SLEEP_SECS=300                       # 5m, for unrecognized non-zero exits
IDLE_SLEEP_SECS=60                         # brief pause between clean-but-unfinished runs
MAX_RESTARTS=200                           # safety valve against infinite loops

COMPLETION_MARKER="ALL_EXPERIMENTS_COMPLETE"

# Set to a non-empty string to pass --dangerously-skip-permissions.
# Leaving this on is what makes *unattended* operation possible (no prompts
# waiting on a human to approve a tool call) — but it means Claude Code can
# take any action without asking. Only use this if you're comfortable with
# that tradeoff for this workdir.
SKIP_PERMISSIONS="1"

# -----------------------------------------------------------------------

mkdir -p "$LOGDIR"

PERM_FLAG=()
if [[ -n "$SKIP_PERMISSIONS" ]]; then
  PERM_FLAG=(--dangerously-skip-permissions)
fi

# Prevent two copies of this wrapper running at once.
exec 200>"$LOCKFILE"
if ! flock -n 200; then
  echo "[$(date)] Another instance is already running (lock held). Exiting."
  exit 1
fi

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/autoresume.log"
}

restart_count=0
first_run=true

log "=== claude_autoresume starting (workdir=$WORKDIR) ==="

while (( restart_count < MAX_RESTARTS )); do
  ts=$(date +%Y%m%d_%H%M%S)
  runlog="$LOGDIR/run_${ts}.log"

  cd "$WORKDIR" || { log "FATAL: cannot cd to $WORKDIR"; exit 1; }

  if $first_run; then
    log "Launch #$restart_count — first run, sending full prompt."
    claude -p "$(cat "$PROMPT_FILE")" "${PERM_FLAG[@]}" 2>&1 | tee "$runlog"
    exit_code=${PIPESTATUS[0]}
    first_run=false
  else
    log "Launch #$restart_count — resuming with --continue."
    claude --continue -p "Resume per the resumption protocol at the top of the prompt and in RUN_STATE.md: read RUN_STATE.md, reconcile real server/process/job state before assuming anything, then continue with the next unfinished step. If you hit a genuine ambiguity, apply the ambiguity-when-unattended rule instead of waiting." \
      "${PERM_FLAG[@]}" 2>&1 | tee "$runlog"
    exit_code=${PIPESTATUS[0]}
  fi

  # ---- Real completion? ----
  if grep -q "$COMPLETION_MARKER" "$runlog"; then
    log "Completion marker found. All experiments reported done. Stopping."
    break
  fi

  # ---- Rate/usage limit hit? ----
  if grep -qiE "rate limit|usage limit|429|exceeded your" "$runlog"; then
    log "Rate/usage limit detected. Sleeping ${DEFAULT_SLEEP_SECS}s (~5h10m) before retry."
    sleep "$DEFAULT_SLEEP_SECS"
    restart_count=$((restart_count+1))
    continue
  fi

  # ---- Any other non-zero exit (crash, network blip, auth error, etc.) ----
  if [[ "$exit_code" -ne 0 ]]; then
    log "Non-zero exit ($exit_code), not a recognized rate-limit message. Sleeping ${CRASH_SLEEP_SECS}s and retrying."
    sleep "$CRASH_SLEEP_SECS"
    restart_count=$((restart_count+1))
    continue
  fi

  # ---- Clean exit but no completion marker: Claude just stopped talking ----
  log "Clean exit without completion marker — task not finished. Retrying shortly."
  sleep "$IDLE_SLEEP_SECS"
  restart_count=$((restart_count+1))
done

if (( restart_count >= MAX_RESTARTS )); then
  log "Hit MAX_RESTARTS ($MAX_RESTARTS) without seeing the completion marker. Stopping — this needs a human look, something is probably stuck or looping."
fi

log "=== claude_autoresume exiting ==="

# -----------------------------------------------------------------------
# OPTIONAL: run this as a systemd user service instead of nohup, so it also
# survives your terminal/SSH session closing and can be checked with
# `systemctl status`:
#
#   ~/.config/systemd/user/claude-autoresume.service
#   ---------------------------------------------
#   [Unit]
#   Description=Claude Code auto-resume wrapper
#
#   [Service]
#   ExecStart=/bin/bash /path/to/claude_autoresume.sh
#   Restart=on-failure
#   RestartSec=30
#
#   [Install]
#   WantedBy=default.target
#   ---------------------------------------------
#
#   systemctl --user daemon-reload
#   systemctl --user enable --now claude-autoresume.service
#   systemctl --user status claude-autoresume.service
#   journalctl --user -u claude-autoresume.service -f
# -----------------------------------------------------------------------
