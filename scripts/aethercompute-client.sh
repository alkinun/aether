#!/usr/bin/env bash
#
# aethercompute-client.sh — one-command volunteer launcher.
#
# This script is a *self-installing* wrapper. It sandboxes everything it needs
# under `<repo>/.aethercompute/` (rust toolchain, cargo registry, a Python venv
# with torch) so your global system is never touched, then builds the branded
# `aether-volunteer` TUI and hands the terminal over to it. The TUI performs
# onboarding, compiles the real training client with live progress, and execs
# it when you're ready.
#
# Usage:
#   ./scripts/aethercompute-client.sh            # install (if needed) + launch
#   ./scripts/aethercompute-client.sh uninstall  # wipe the sandbox
#   ./scripts/aethercompute-client.sh doctor     # show what's installed/missing
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SANDBOX="$REPO_ROOT/.aethercompute"
export RUSTUP_HOME="$SANDBOX/rustup"
export CARGO_HOME="$SANDBOX/cargo"
VENV="$SANDBOX/venv"
VOLUNTEER_BIN="$REPO_ROOT/target/release/aether-volunteer"
LOG_DIR="$SANDBOX/install-logs"
INSTALL_LOG="$LOG_DIR/install.log"

VOLUNTEER_CRATE="psyche-centralized-volunteer"
RUST_PROFILE="minimal"
# Fallback torch version for hosts with NO system torch. The pinned tch-rs needs
# a libtorch build that pip wheels don't reliably provide (symbols move between
# minor torch releases), so we prefer the user's existing system torch and only
# install this as a last resort. 2.11.0 is empirically compatible with this rev.
TORCH_VERSION="2.11.0"

# --- pretty output ---------------------------------------------------------
# Keep this palette aligned with architectures/centralized/volunteer/src/brand.rs.
bold="\033[1m"; reset="\033[0m"
rose="\033[38;2;218;78;138m"       # BRAND_A / danger
cyan="\033[38;2;82;184;205m"       # BRAND_B / success
amber="\033[38;2;226;136;68m"      # warn
bone="\033[38;2;226;204;184m"      # ink
dim="\033[38;2;116;98;104m"        # muted text

brand() { printf "${rose}${bold}%s${reset}" "$1"; }

ok()    { printf "  ${cyan}✓${reset}  %s\n" "$1"; }
warn()  { printf "  ${amber}!${reset}  %s\n" "$1"; }
fail()  { printf "  ${rose}✗${reset}  %s\n" "$1"; }
hint()  { printf "    ${dim}↳ %s${reset}\n" "$1"; }
die()   { fail "$1"; exit 1; }

# run_step "<label>" <command...>
# Runs a command with a spinner, streaming output to $INSTALL_LOG.
run_step() {
  local label="$1"; shift
  mkdir -p "$LOG_DIR"
  : > "$INSTALL_LOG"
  ("$@" >>"$INSTALL_LOG" 2>&1) &
  local pid=$!
  spin "$pid" "$label"
  if wait "$pid"; then
    ok "$label"
    return 0
  else
    fail "$label"
    tail -n 20 "$INSTALL_LOG" >&2 || true
    return 1
  fi
}

spin() {
  local pid=$1 label=$2
  local frames=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
  local i=0
  while kill -0 "$pid" 2>/dev/null; do
    printf "\r\033[K  ${cyan}${bold}%s${reset}  ${dim}%s${reset}" "${frames[$((i % ${#frames[@]}))]}" "$label"
    i=$((i + 1))
    sleep 0.08
  done
  printf "\r\033[K"
}

# --- platform / capability detection ---------------------------------------
is_macos()  { [[ "$(uname -s)" == "Darwin" ]]; }
is_linux()  { [[ "$(uname -s)" == "Linux" ]]; }
has_nvidia() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; }

has() { command -v "$1" >/dev/null 2>&1; }

# --- individual setup steps ------------------------------------------------
ensure_dirs() { mkdir -p "$SANDBOX" "$LOG_DIR"; }

ensure_rust() {
  if [[ -x "$CARGO_HOME/bin/cargo" ]]; then return 0; fi
  if ! has curl; then die "curl is required to bootstrap rust. Please install it and re-run."; fi

  local installer
  installer="$(mktemp)"
  run_step "downloading rustup-init" \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o "$installer" \
    || { rm -f "$installer"; die "could not download rustup-init. Are you online?"; }
  run_step "installing rust toolchain (sandboxed)" \
    sh "$installer" -y --no-modify-path --profile "$RUST_PROFILE" --default-toolchain stable \
    || { rm -f "$installer"; die "rustup install failed. See $INSTALL_LOG"; }
  rm -f "$installer"
  "$CARGO_HOME/bin/rustup" default stable >>"$INSTALL_LOG" 2>&1 || true
}

ensure_c_compiler() {
  if has cc || has gcc || has clang; then return 0; fi
  warn "no C compiler found (cc/gcc/clang)."
  if is_macos; then
    hint "install with: xcode-select --install"
  else
    hint "install with: sudo apt-get install -y build-essential"
  fi
  die "a C compiler is required to build the training client."
}

ensure_python() {
  if [[ -x "$VENV/bin/python" ]]; then return 0; fi
  if ! has python3; then
    warn "python3 not found."
    if is_macos; then
      hint "install with: xcode-select --install   (or: brew install python)"
    else
      hint "install with: sudo apt-get install -y python3 python3-venv"
    fi
    die "python3 + venv is required to provision libtorch."
  fi
  if ! python3 -m venv --help >/dev/null 2>&1; then
    warn "python venv module missing."
    is_linux && hint "install with: sudo apt-get install -y python3-venv"
    die "python3-venv is required."
  fi
  run_step "creating sandboxed python venv" python3 -m venv "$VENV" || die "venv creation failed"
  run_step "upgrading pip" "$VENV/bin/python" -m pip install --upgrade pip
}

ensure_torch() {
  # tch-rs needs a libtorch build that pip wheels don't reliably provide (symbols
  # move between minor torch releases). So prefer the user's existing system
  # torch — that's the combination known to work for their build.
  if python3 -c 'import torch' >/dev/null 2>&1; then
    local ver
    ver="$(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null || echo "?")"
    ok "using system torch ${ver}"
    return 0
  fi

  # No system torch — provision one in the sandbox as a best-effort fallback.
  warn "no system torch found — installing $TORCH_VERSION into the sandbox"
  ensure_python || return 1

  if is_linux && ! has_nvidia; then
    run_step "installing torch $TORCH_VERSION (CPU)" \
      "$VENV/bin/pip" install "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cpu \
      || die "torch install failed. See $INSTALL_LOG"
  else
    local flavor="CUDA"
    is_macos && flavor="macOS (MPS-capable)"
    run_step "installing torch $TORCH_VERSION ($flavor)" \
      "$VENV/bin/pip" install "torch==$TORCH_VERSION" \
      || die "torch install failed. See $INSTALL_LOG"
  fi
}

torch_lib_dirs() {
  # torch/lib plus every nvidia/*/lib from the CUDA pip wheels — all needed on
  # the loader path for libtorch_cuda to resolve. Uses the system python3
  # (matching the torch the build/run actually uses).
  python3 -c '
import pathlib
try:
    import torch
except Exception:
    raise SystemExit(0)
tf = pathlib.Path(torch.__file__).resolve()
dirs = [str(tf.parent / "lib")]
nv = tf.parent.parent / "nvidia"
if nv.is_dir():
    dirs += [str(d) for d in sorted(nv.glob("*/lib"))]
print(":".join(dirs))
' 2>/dev/null
}

ensure_volunteer_bin() {
  if [[ -x "$VOLUNTEER_BIN" ]]; then
    # Skip the rebuild only when no source file (or Cargo.toml) is newer than
    # the binary — otherwise a stale launcher hides UI changes from the user.
    local src="$REPO_ROOT/architectures/centralized/volunteer"
    if ! find "$src/src" "$src/Cargo.toml" -type f -newer "$VOLUNTEER_BIN" 2>/dev/null | grep -q .; then
      return 0
    fi
    warn "launcher source changed — rebuilding"
  fi
  run_step "compiling the aether-volunteer launcher" "$CARGO_HOME/bin/cargo" build --release -p "$VOLUNTEER_CRATE" \
    || die "volunteer build failed. See $INSTALL_LOG"
}

# --- subcommands -----------------------------------------------------------
do_uninstall() {
  printf "  ${amber}Removing sandbox %s${reset}\n" "$SANDBOX"
  rm -rf "$SANDBOX"
  ok "sandbox removed (repo source untouched)."
}

do_doctor() {
  printf "  %s\n\n" "$(brand '◆ AETHERCOMPUTE · environment check')"
  check "sandbox dir"        "[[ -d '$SANDBOX' ]]"
  check "cargo (sandbox)"    "[[ -x '$CARGO_HOME/bin/cargo' ]]"
  check "rust toolchain"     "'$CARGO_HOME/bin/rustc' -V >/dev/null 2>&1"
  check "C compiler"         "command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1"
  check "python3"            "command -v python3 >/dev/null 2>&1"
  check "system torch"       "python3 -c 'import torch' >/dev/null 2>&1"
  check "launcher binary"    "[[ -x '$VOLUNTEER_BIN' ]]"
  echo
}

check() {
  local label="$1" test="$2"
  if eval "$test" >/dev/null 2>&1; then ok "$label"; else fail "$label"; fi
}

do_launch() {
  ensure_dirs
  # Boot animation while we figure out what's missing.
  printf "\n  "; brand '◆ AETHERCOMPUTE'; printf "  ${dim}volunteer launcher${reset}\n\n"

  ensure_rust || exit 1
  ensure_c_compiler || exit 1
  ensure_torch || exit 1
  ensure_volunteer_bin || exit 1

  local torch_libs
  torch_libs="$(torch_lib_dirs)"
  export LD_LIBRARY_PATH="${torch_libs:+$torch_libs:}${LD_LIBRARY_PATH:-}"
  export DYLD_LIBRARY_PATH="${torch_libs:+$torch_libs:}${DYLD_LIBRARY_PATH:-}"
  export LIBTORCH_USE_PYTORCH=1
  export LIBTORCH_BYPASS_VERSION_CHECK=1
  export RUST_MIN_STACK=268435456
  export PATH="$CARGO_HOME/bin:$PATH"

  printf "  ${cyan}${bold}setup complete${reset}\n"
  printf "  ${dim}handing off to the launcher…${reset}\n\n"
  exec "$VOLUNTEER_BIN" "$@"
}

main() {
  case "${1:-}" in
    uninstall) do_uninstall; exit 0 ;;
    doctor)    ensure_dirs; do_doctor; exit 0 ;;
    -h|--help|help)
      sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    "") do_launch "$@" ;;
    *) printf "${rose}unknown subcommand: %s${reset}\n" "$1" >&2; exit 2 ;;
  esac
}

main "$@"
