#!/bin/sh
#
# aethercompute-seed.sh — seed node launcher.
#
# The seed node trains AND pushes model checkpoints to HuggingFace Hub every
# epoch. Volunteer nodes (launched via aethercompute-client.sh) download these
# checkpoints and join training without needing HF write credentials.
#
# Usage:
#   ./scripts/aethercompute-seed.sh                              # dev mode
#   curl -fsSL https://aethercompute.org/seed.sh | HF_TOKEN=... HUB_REPO=... sh
#
# Required environment:
#   HF_TOKEN   HuggingFace access token with write access
#   HUB_REPO   target repo, e.g. "user/model-name"
#
# This is a thin wrapper around aethercompute-client.sh that sets checkpoint
# environment variables before delegating. See --help on that script for more.
# -----------------------------------------------------------------------------
exec sh "$(dirname "$0")/aethercompute-client.sh" seed "$@"
