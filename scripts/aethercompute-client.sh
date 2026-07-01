#!/bin/sh
#
# aethercompute-client.sh — one-command volunteer launcher.
#
# Fetches the aether source (cloning it under ~/.aethercompute on first run),
# sandboxes a whole toolchain (rustup, cargo registry, a python venv + torch)
# under <repo>/.aethercompute so the global system is never touched, builds the
# aether-volunteer TUI, and hands the terminal over to it. The TUI performs
# onboarding, compiles the real training client with live progress, and execs
# it when you're ready.
#
# Usage:
#   curl -fsSL https://aethercompute.org/client.sh | sh            # volunteer node
#   curl -fsSL https://aethercompute.org/client.sh | sh -s seed    # seed node (requires HF_TOKEN, HUB_REPO)
#   curl -fsSL https://aethercompute.org/client.sh | sh -s update  # pull latest source
#   curl -fsSL https://aethercompute.org/client.sh | sh -s doctor  # env check
#   curl -fsSL https://aethercompute.org/client.sh | sh -s uninstall
#
# This script targets bash. When piped to a POSIX `sh` (e.g. dash on Debian),
# the bootstrap below re-fetches it and runs it under bash automatically.
# -----------------------------------------------------------------------------

# --- bash bootstrap ---------------------------------------------------------
if [ -z "${BASH_VERSION:-}" ]; then
  _installer="${AETHER_INSTALLER_URL:-https://aethercompute.org/client.sh}"
  command -v bash >/dev/null 2>&1 || {
    echo "aethercompute: bash is required to run the installer." >&2; exit 1; }
  _tmp="$(mktemp 2>/dev/null || printf '/tmp/aether-client.%d.sh' "$$")"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$_installer" -o "$_tmp" || _rc=1
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$_tmp" "$_installer" || _rc=1
  else
    echo "aethercompute: curl or wget is required." >&2; exit 1
  fi
  if [ "${_rc:-0}" -ne 0 ]; then
    rm -f "$_tmp"
    echo "aethercompute: could not download installer from $_installer" >&2
    exit 1
  fi
  bash "$_tmp" "$@"; _rc=$?; rm -f "$_tmp"; exit "$_rc"
fi

# ===== bash-only below ======================================================
set -euo pipefail

# --- config (overridable via environment) -----------------------------------
AETHER_HOME="${AETHER_HOME:-$HOME/.aethercompute}"
REPO_URL="${AETHER_REPO_URL:-https://github.com/alkinun/aether.git}"
REPO_REF="${AETHER_REPO_REF:-main}"

VOLUNTEER_CRATE="psyche-centralized-volunteer"
RUST_PROFILE="minimal"
# Pinned torch used by torch-sys/tch. Do not mix arbitrary system torch with this
# build: minor torch releases change C++ symbols and can fail at link time.
TORCH_VERSION="2.9.1"

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
is_macos()   { [[ "$(uname -s)" == "Darwin" ]]; }
is_linux()   { [[ "$(uname -s)" == "Linux" ]]; }
has_nvidia() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; }

has() { command -v "$1" >/dev/null 2>&1; }

# --- path resolution -------------------------------------------------------
# Dev mode:  the script lives at <repo>/scripts/aethercompute-client.sh, so the
#            repo is one level up from the script's own directory.
# Standalone (curl | sh): BASH_SOURCE is empty/stdin, so there is no local repo;
#            the source is cloned under AETHER_HOME and treated as the repo root.
resolve_paths() {
  local self="${BASH_SOURCE[0]:-}"
  if [[ -n "$self" && -f "$self" ]]; then
    local sdir
    sdir="$(cd "$(dirname "$self")" 2>/dev/null && pwd)" || sdir=""
    if [[ -n "$sdir" && -f "$sdir/../Cargo.toml" ]]; then
      REPO_ROOT="$(cd "$sdir/.." && pwd)"
      EMBEDDED_REPO=1
      setup_paths
      return
    fi
  fi
  REPO_ROOT="$AETHER_HOME/repo"
  EMBEDDED_REPO=0
  setup_paths
}

# All sandboxed state lives under <repo>/.aethercompute — this MUST match
# `sandbox_dir()` in architectures/centralized/volunteer/src/config.rs, which
# derives it from the crate's compile-time CARGO_MANIFEST_DIR. The install logs
# live under AETHER_HOME (not the sandbox) so they exist before the first clone.
setup_paths() {
  SANDBOX="$REPO_ROOT/.aethercompute"
  export RUSTUP_HOME="$SANDBOX/rustup"
  export CARGO_HOME="$SANDBOX/cargo"
  VENV="$SANDBOX/venv"
  VOLUNTEER_BIN="$REPO_ROOT/target/release/aether-volunteer"
  LOG_DIR="$AETHER_HOME/install-logs"
  INSTALL_LOG="$LOG_DIR/install.log"
}

# --- source acquisition ----------------------------------------------------
ensure_dirs() { mkdir -p "$SANDBOX" "$LOG_DIR"; }

ensure_repo() {
  if [[ "$EMBEDDED_REPO" == "1" ]]; then return 0; fi
  mkdir -p "$AETHER_HOME"
  if [[ -f "$REPO_ROOT/Cargo.toml" ]]; then return 0; fi
  if has git; then
    run_step "fetching aether source" \
      git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$REPO_ROOT" \
      || die "could not clone $REPO_URL. See $INSTALL_LOG"
  elif has tar; then
    fetch_tarball
  else
    die "need 'git' or 'tar' to fetch the aether source. Install one and re-run."
  fi
}

# Tarball fallback when git is unavailable. GitHub serves source archives at
# <repo>/archive/<ref>.tar.gz (works for branches and tags).
fetch_tarball() {
  local archive tarball_url="${REPO_URL%.git}/archive/${REPO_REF}.tar.gz"
  archive="$(mktemp)"
  run_step "downloading aether source" \
    curl -fsSL "$tarball_url" -o "$archive" \
    || { rm -f "$archive"; die "could not download source from $tarball_url. See $INSTALL_LOG"; }
  mkdir -p "$REPO_ROOT"
  run_step "extracting aether source" \
    tar -xzf "$archive" -C "$REPO_ROOT" --strip-components=1 \
    || { rm -f "$archive"; die "could not extract source archive. See $INSTALL_LOG"; }
  rm -f "$archive"
}

# --- individual setup steps ------------------------------------------------
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
  ensure_python || return 1

  if "$VENV/bin/python" -c "import torch, sys; sys.exit(0 if torch.__version__.split('+', 1)[0] == '$TORCH_VERSION' else 1)" >/dev/null 2>&1; then
    local ver
    ver="$("$VENV/bin/python" -c 'import torch;print(torch.__version__)' 2>/dev/null || echo "?")"
    ok "using sandbox torch ${ver}"
    return 0
  fi

  warn "installing sandbox torch $TORCH_VERSION"

  if is_linux && ! has_nvidia; then
    run_step "installing torch $TORCH_VERSION (CPU)" \
      "$VENV/bin/pip" install --force-reinstall "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cpu \
      || die "torch install failed. See $INSTALL_LOG"
  else
    local flavor="CUDA"
    is_macos && flavor="macOS (MPS-capable)"
    run_step "installing torch $TORCH_VERSION ($flavor)" \
      "$VENV/bin/pip" install --force-reinstall "torch==$TORCH_VERSION" \
      || die "torch install failed. See $INSTALL_LOG"
  fi
}

torch_lib_dirs() {
  # torch/lib plus every nvidia/*/lib from the CUDA pip wheels — all needed on
  # the loader path for libtorch_cuda to resolve. Uses the sandbox python so the
  # build and runtime libtorch always match.
  "$VENV/bin/python" -c '
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
show_help() {
  cat <<'HELP'
aethercompute-client.sh — one-command volunteer launcher.

Fetches the aether source, sandboxes a whole toolchain (rustup, cargo registry,
python venv + torch) under ~/.aethercompute, builds the aether-volunteer TUI,
and hands the terminal over to it. The TUI performs onboarding, compiles the
real training client with live progress, and execs it when you're ready.

Usage:
  curl -fsSL https://aethercompute.org/client.sh | sh           volunteer node
  curl -fsSL https://aethercompute.org/client.sh | sh -s seed   seed node (requires HF_TOKEN, HUB_REPO)
  curl -fsSL https://aethercompute.org/client.sh | sh -s update pull latest source
  curl -fsSL https://aethercompute.org/client.sh | sh -s doctor show what's installed
  curl -fsSL https://aethercompute.org/client.sh | sh -s uninstall

Subcommands:
  (none)    volunteer node: train without uploading checkpoints
  seed      seed node: train and push checkpoints to HuggingFace Hub every epoch
  update    fetch the latest aether source (re-run without args to rebuild)
  doctor    diagnose the local environment
  uninstall remove all aethercompute data

Seed mode environment (required):
  HF_TOKEN     HuggingFace access token with write access
  HUB_REPO     target repo, e.g. "user/model-name"

Seed mode environment (optional):
  CHECKPOINT_DIR             local checkpoint storage (default: ~/.aethercompute/checkpoints)
  CHECKPOINT_EPOCH_INTERVAL  push every N epochs (default: 1)
  KEEP_STEPS                 step checkpoints to retain (default: 3)

Environment:
  AETHER_HOME          install root (default: ~/.aethercompute)
  AETHER_REPO_URL      git source to clone (default: github.com/alkinun/aether)
  AETHER_REPO_REF      branch/tag to use (default: main)
  AETHER_INSTALLER_URL self-URL for the POSIX sh -> bash re-exec
HELP
}

check() {
  local label="$1" test="$2"
  if eval "$test" >/dev/null 2>&1; then ok "$label"; else fail "$label"; fi
}

do_doctor() {
  printf "  %s\n\n" "$(brand '◆ AETHERCOMPUTE · environment check')"
  check "source repo"      "[[ -f '$REPO_ROOT/Cargo.toml' ]]"
  check "sandbox dir"      "[[ -d '$SANDBOX' ]]"
  check "cargo (sandbox)"  "[[ -x '$CARGO_HOME/bin/cargo' ]]"
  check "rust toolchain"   "'$CARGO_HOME/bin/rustc' -V >/dev/null 2>&1"
  check "C compiler"       "command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 || command -v clang >/dev/null 2>&1"
  check "python3"          "command -v python3 >/dev/null 2>&1"
  check "system torch"     "python3 -c 'import torch' >/dev/null 2>&1"
  check "launcher binary"  "[[ -x '$VOLUNTEER_BIN' ]]"
  echo
}

do_uninstall() {
  if [[ "$EMBEDDED_REPO" == "1" ]]; then
    printf "  ${amber}Removing sandbox %s${reset}\n" "$SANDBOX"
    rm -rf "$SANDBOX"
    ok "sandbox removed (repo source untouched)."
  else
    printf "  ${amber}Removing %s${reset}\n" "$AETHER_HOME"
    rm -rf "$AETHER_HOME"
    ok "all aethercompute data removed."
  fi
}

do_update() {
  printf "\n  "; brand '◆ AETHERCOMPUTE'; printf "  ${dim}update${reset}\n\n"
  if has git && [[ -d "$REPO_ROOT/.git" ]]; then
    run_step "pulling latest source" \
      git -C "$REPO_ROOT" pull --ff-only \
      || ( warn "fast-forward failed; forced update detected, resetting to remote..." \
          && git -C "$REPO_ROOT" reset --hard "@{upstream}" \
          && ok "source reset to remote." \
        ) \
      || warn "could not pull updates (continuing with existing source)."
  else
    # Tarball install (or embedded repo without git): re-fetch from scratch.
    rm -rf "$REPO_ROOT"
    if has git; then
      mkdir -p "$AETHER_HOME"
      run_step "re-cloning aether source" \
        git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$REPO_ROOT" \
        || die "could not clone $REPO_URL. See $INSTALL_LOG"
    elif has tar; then
      fetch_tarball
    else
      die "need 'git' or 'tar' to update the aether source."
    fi
  fi
  local commit
  commit="$(git -C "$REPO_ROOT" log --oneline -1 2>/dev/null || echo "unknown")"
  printf "  ${dim}now at: ${cyan}%s${reset}\n" "$commit"
  hint "re-run without arguments to rebuild + launch."
}

do_seed() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    die "HF_TOKEN is required for seed mode. Get one at https://huggingface.co/settings/tokens"
  fi
  if [[ -z "${HUB_REPO:-}" ]]; then
    die "HUB_REPO is required for seed mode (e.g. 'user/model-name')"
  fi

  export CHECKPOINT_EPOCH_INTERVAL="${CHECKPOINT_EPOCH_INTERVAL:-1}"
  export KEEP_STEPS="${KEEP_STEPS:-3}"
  export DELETE_OLD_STEPS="${DELETE_OLD_STEPS:-true}"
  export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$AETHER_HOME/checkpoints}"

  printf "\n  "; brand '◆ AETHERCOMPUTE'; printf "  ${dim}seed node${reset}\n\n"
  hint "Hub repo:        $HUB_REPO"
  hint "Checkpoint dir:  $CHECKPOINT_DIR"
  hint "Push interval:   every $CHECKPOINT_EPOCH_INTERVAL epoch(s)"
  hint "Keep steps:      $KEEP_STEPS"
  echo

  do_launch "$@"
}

do_launch() {
  ensure_repo
  ensure_dirs
  cd "$REPO_ROOT"

  # Boot animation while we figure out what's missing.
  printf "\n  "; brand '◆ AETHERCOMPUTE'; printf "  ${dim}volunteer launcher${reset}\n\n"

  ensure_rust || exit 1
  ensure_c_compiler || exit 1
  ensure_torch || exit 1
  ensure_volunteer_bin || exit 1

  # Log the commit hash so users can verify which version they're running.
  local commit
  commit="$(git -C "$REPO_ROOT" log --oneline -1 2>/dev/null || echo "unknown")"
  printf "  ${dim}source: ${cyan}%s${reset}\n" "$commit"

  local torch_libs
  torch_libs="$(torch_lib_dirs)"
  export LD_LIBRARY_PATH="${torch_libs:+$torch_libs:}${LD_LIBRARY_PATH:-}"
  export DYLD_LIBRARY_PATH="${torch_libs:+$torch_libs:}${DYLD_LIBRARY_PATH:-}"
  export LIBTORCH_USE_PYTORCH=1
  export LIBTORCH_BYPASS_VERSION_CHECK=1
  export RUST_MIN_STACK=268435456
  export VIRTUAL_ENV="$VENV"
  export PATH="$VENV/bin:$CARGO_HOME/bin:$PATH"

  printf "  ${cyan}${bold}setup complete${reset}\n"
  printf "  ${dim}handing off to the launcher…${reset}\n\n"
  exec "$VOLUNTEER_BIN" "$@"
}

main() {
  resolve_paths
  case "${1:-}" in
    seed)      shift; do_seed "$@" ;;
    update)    do_update; exit 0 ;;
    uninstall) do_uninstall; exit 0 ;;
    doctor)    do_doctor; exit 0 ;;
    -h|--help|help) show_help; exit 0 ;;
    "")        do_launch "$@" ;;
    *) printf "${rose}unknown subcommand: %s${reset}\n" "$1" >&2
       show_help >&2; exit 2 ;;
  esac
}

main "$@"
