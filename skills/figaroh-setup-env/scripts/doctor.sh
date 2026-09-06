#!/usr/bin/env bash
# FIGAROH environment doctor — read-only checks, no mutations.
#
# Usage:  bash doctor.sh [--ws /path/to/figaroh-ws]
# Exit:   0 = ready to run, 1 = at least one blocking problem
#
# Every FAIL line is followed by the exact remedy. Optional deps report WARN,
# not FAIL: they only block specific tasks (named in the message).

set -uo pipefail

# Workspace root = the directory holding `figaroh/` and `figaroh-examples/` side by
# side. Resolved in order: $FIGAROH_WS / $WS, else walk up from this script (it ships
# inside <repo>/skills/figaroh-setup-env/scripts/), else the current directory.
_find_ws() {
  local d
  d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  while [ "$d" != "/" ]; do
    if [ -d "$d/figaroh" ] && [ -d "$d/figaroh-examples" ]; then echo "$d"; return 0; fi
    d="$(dirname "$d")"
  done
  d="$PWD"
  while [ "$d" != "/" ]; do
    if [ -d "$d/figaroh" ] && [ -d "$d/figaroh-examples" ]; then echo "$d"; return 0; fi
    d="$(dirname "$d")"
  done
  return 1
}
WS="${FIGAROH_WS:-${WS:-$(_find_ws)}}"
ENV_NAME="figaroh-dev"
while [ $# -gt 0 ]; do
  case "$1" in
    --ws) WS="$2"; shift 2 ;;
    --env) ENV_NAME="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

FAILED=0
pass() { printf '  \033[0;32mPASS\033[0m  %s\n' "$1"; }
warn() { printf '  \033[1;33mWARN\033[0m  %s\n' "$1"; }
fail() { printf '  \033[0;31mFAIL\033[0m  %s\n' "$1"; FAILED=1; }
fix()  { printf '        fix: %s\n' "$1"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$1"; }

head_ "1. Workspace layout"
if [ -z "$WS" ]; then
  fail "cannot locate the workspace root (a directory containing both figaroh/ and figaroh-examples/)"
  fix "re-run with --ws <root>, or export FIGAROH_WS=<root>"
  printf '\n  \033[0;31mBlocked.\033[0m\n\n'; exit 1
fi
printf '  using workspace root: %s\n' "$WS"
for d in figaroh figaroh-examples; do
  if [ -d "$WS/$d" ]; then pass "$WS/$d"
  else fail "missing $WS/$d"; fix "clone it next to the other repos, or re-run with --ws <root>"; fi
done
[ -f "$WS/figaroh/pyproject.toml" ] || { fail "no pyproject.toml in $WS/figaroh"; fix "wrong --ws root?"; }
[ -d "$WS/figaroh-examples/models" ] && pass "shared models/ present" \
  || warn "no $WS/figaroh-examples/models — meshes will not resolve (package_dirs=../../models)"

head_ "2. Conda environment"
if ! command -v conda >/dev/null 2>&1; then
  fail "conda not on PATH"; fix "install miniforge, then re-open the shell"
elif conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  pass "conda env '$ENV_NAME' exists"
else
  fail "conda env '$ENV_NAME' does not exist"
  fix "conda env create -f $WS/figaroh/environment.yml"
fi

RUN="conda run -n $ENV_NAME"

# `conda run` appends a trailing blank line to captured stdout, so `tail -1`
# yields "" instead of the result. Every capture below prints a sentinel-
# prefixed line and greps for it.
probe() { $RUN python -c "$1" 2>/dev/null | sed -n 's/^__FIG__ //p' | head -1; }

head_ "3. Python + figaroh import"
PY_VER=$(probe 'import sys;print("__FIG__ %d.%d.%d"%sys.version_info[:3])')
if [ -n "$PY_VER" ]; then pass "python $PY_VER in '$ENV_NAME'"
else fail "cannot run python in '$ENV_NAME'"; fix "conda env create -f $WS/figaroh/environment.yml"; fi

FIG=$(probe 'import figaroh;print("__FIG__", figaroh.__version__, figaroh.__file__)')
if [ -n "$FIG" ]; then
  pass "import figaroh -> $FIG"
  case "$FIG" in
    *"$WS/figaroh/src/figaroh"*) pass "editable install points at the local source tree" ;;
    *) warn "figaroh does NOT resolve to this workspace's source tree ($WS/figaroh/src)"
       warn "  it may be a site-packages copy or another checkout — edits made here will be ignored"
       fix "conda run -n $ENV_NAME pip install -e $WS/figaroh" ;;
  esac
else
  fail "import figaroh failed"
  fix "conda run -n $ENV_NAME pip install -e $WS/figaroh"
fi

head_ "4. Core dependencies"
for mod in pinocchio numpy scipy pandas yaml matplotlib; do
  if $RUN python -c "import $mod" >/dev/null 2>&1; then pass "$mod"
  else fail "$mod missing"; fix "conda run -n $ENV_NAME pip install -e $WS/figaroh"; fi
done

head_ "5. Optional dependencies (each blocks one task only)"
if $RUN python -c 'import cyipopt' >/dev/null 2>&1; then pass "cyipopt — optimal_trajectory available"
else warn "cyipopt missing — blocks figaroh.optimal / optimal_trajectory.py ONLY"
     fix "conda install -n $ENV_NAME -c conda-forge cyipopt   (not pip-installable)"; fi
if $RUN python -c 'import picos' >/dev/null 2>&1; then pass "picos — SDP physical-consistency projection available"
else warn "picos missing — blocks identification.physical_consistency (default-off)"
     fix "conda run -n $ENV_NAME pip install picos cvxopt"; fi
for m in viser meshcat; do
  $RUN python -c "import $m" >/dev/null 2>&1 && pass "$m — visualisation available" \
    || warn "$m missing — blocks --viz-validation / 3D preview only"
done

head_ "6. Smoke test (UR10 reference example)"
UR="$WS/figaroh-examples/examples/ur10"
if [ -f "$UR/calibration.py" ] && [ -f "$UR/config/ur10_unified_config.yaml" ] && [ -f "$UR/urdf/ur10_robot.urdf" ]; then
  pass "UR10 reference example is complete"
  printf '        run: cd %s && conda run -n %s python calibration.py --calibrate-only --no-plot\n' "$UR" "$ENV_NAME"
else
  fail "UR10 reference example incomplete under $UR"
  fix "git status in $WS/figaroh-examples — the reference example must be intact"
fi

head_ "Result"
if [ "$FAILED" -eq 0 ]; then
  printf '  \033[0;32mReady.\033[0m Activate with: conda activate %s\n\n' "$ENV_NAME"
else
  printf '  \033[0;31mBlocked.\033[0m Apply the fix lines above, then re-run this script.\n\n'
fi
exit $FAILED
