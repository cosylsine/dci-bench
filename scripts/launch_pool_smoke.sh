#!/usr/bin/env bash
set -Eeuo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$repo/releases/protocol-freezes/20261026-pool-smoke-v4/launch.sh" "$@"
