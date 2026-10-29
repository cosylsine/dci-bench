#!/usr/bin/env bash
# Cluster images may omit both tools. Keep package-manager libraries separate
# from Conda; the benchmark requires trusted binaries at these exact paths.
set -Eeuo pipefail
missing=()
[[ -x /usr/bin/bwrap ]] || missing+=(bubblewrap)
[[ -x /usr/bin/rg ]] || missing+=(ripgrep)
if (( ${#missing[@]} )); then
  echo "[smoke] Installing missing system dependencies: ${missing[*]}"
  if [[ "$(id -u)" == 0 ]]; then
    elevate=()
  elif command -v sudo >/dev/null && sudo -n true 2>/dev/null; then
    elevate=(sudo -n)
  else
    echo '[smoke] Missing bubblewrap/ripgrep and no root or passwordless sudo; request a cluster image with these system packages.' >&2
    exit 2
  fi
  if command -v apt-get >/dev/null; then
    "${elevate[@]}" env -u LD_LIBRARY_PATH -u LD_PRELOAD \
      apt-get update -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30
    "${elevate[@]}" env -u LD_LIBRARY_PATH -u LD_PRELOAD DEBIAN_FRONTEND=noninteractive \
      apt-get install -y --no-install-recommends -o DPkg::Lock::Timeout=180 \
      -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 "${missing[@]}"
  elif command -v dnf >/dev/null; then
    "${elevate[@]}" env -u LD_LIBRARY_PATH -u LD_PRELOAD dnf install -y "${missing[@]}"
  elif command -v yum >/dev/null; then
    "${elevate[@]}" env -u LD_LIBRARY_PATH -u LD_PRELOAD yum install -y "${missing[@]}"
  else
    echo '[smoke] No supported system package manager (apt-get/dnf/yum).' >&2
    exit 2
  fi
fi
for executable in /usr/bin/bwrap /usr/bin/rg; do
  [[ -x "$executable" ]] || { echo "[smoke] Required executable unavailable after install: $executable" >&2; exit 2; }
  env -u LD_LIBRARY_PATH -u LD_PRELOAD "$executable" --version
done
# Detect container-policy failures before model loading. Full Pi isolation and
# Inspect bridge acceptance still run in the Python preflight afterwards.
if ! env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/bwrap \
    --unshare-user --uid 65534 --gid 65534 --unshare-pid --unshare-net \
    --unshare-ipc --unshare-uts --cap-drop ALL --ro-bind / / \
    --die-with-parent --new-session /bin/true; then
  echo '[smoke] bubblewrap is installed, but the container denied namespace creation. Enable the cluster container namespace/security permissions; GPU/model loading has not started.' >&2
  exit 2
fi
