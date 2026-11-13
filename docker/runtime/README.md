# Runtime build inputs

The Dockerfile pins the Debian 13 amd64 base manifest, Debian snapshot and ripgrep
package. Docker is optional on a build machine and is not used by the evaluation
runner.

The primary daemonless builder is `scripts/build_frozen_runtime.py`. It exports
the pinned base with crane, installs the pinned package in a build-only bwrap
environment, normalizes links and permissions, and creates a new OCI archive
plus the offline rootfs bundle. It never pushes to a registry.

The final runtime digest must come from that final normalized image, rather than
from the base image or an intermediate Docker build. See
[the runtime guide](../../docs/frozen_runtime.md) for commands and validation.
