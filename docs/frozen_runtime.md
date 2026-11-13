# Frozen RootFS runtime

Frozen mode gives the Bash tool a pinned Debian userland instead of the host's
`/usr` directories. Inspect, the model bridge, custom `read`, candidate pools and
scoring remain host-side. It is opt-in; existing host-tools results and the Code
Track retain their existing behavior.

## Prepare once, evaluate offline

Use the benchmark Python environment. The checked-in
`configs/runtime/frozen-debian13.json` specifies the content digests and defaults
to the executable node-local cache `/tmp/dci-bench-runtime`. Relative cache paths
are resolved against the configuration file. Changing this path does not change
the logical Runtime identity.

```bash
python -m dci_bench.runtime prepare \
  --config configs/runtime/frozen-debian13.json \
  --bundle /path/to/release/rootfs.tar --offline
python -m dci_bench.runtime verify --config configs/runtime/frozen-debian13.json
python scripts/check_frozen_runtime.py \
  --runtime-config configs/runtime/frozen-debian13.json
```

The four-5090 launcher performs the same preparation in this order: host
dependency preparation and namespace validation, offline Frozen cache import or
reuse plus the actual no-model Pi probe, data preflight and synthetic bridge
preflight, then GPU startup. Set `DCI_RUNTIME_BUNDLE` to the explicit local
`rootfs.tar` when the node-local cache is cold; a valid warm cache can be reused
without the variable. The launcher passes `--offline --preflight` to the runtime
manager, so a missing cold cache fails rather than downloading or falling back to
host tools. A no-argument `DRY_RUN=1` launch exits before dependency
installation or bundle extraction.

Direct evaluation entrypoints also require a runtime tool preflight when
`--runtime-config` is supplied: they verify the pinned files, then use host
`/usr/bin/bwrap` to execute Bash and `/usr/bin/rg --version` inside the RootFS.
A valid warm cache does not skip this check. Missing Bubblewrap, denied
namespaces, or an unusable RootFS ripgrep stop evaluation before model requests.
Host ripgrep does not satisfy the Frozen check.

The sandbox check defaults to `document-pool-v2-lite`; use
`--profile document-pool-v1-small` for the v1 stable-rg mount policy. Its
`--startup-timeout-seconds` defaults to 240 for Node/source loading and sandbox
startup. This operational guard is independent of sample and Bash budgets;
the original 60-second startup guard was insufficient on this AFS checkout.

Preparation checks the bundle digest, rejects unsafe archive paths/links and
special files, normalizes read-only permissions, verifies the file tree, and
publishes the cache with a lock and atomic READY marker. Missing or corrupt
caches fail closed. Evaluation never downloads or installs a runtime. Existing
published caches are not overwritten or automatically repaired.

The host must have trusted `/usr/bin/bwrap` and the existing Python/Node/Pi
dependencies. Frozen mode does not require host ripgrep. Ubuntu 20.04's bwrap 0.4
is supported through an explicit child environment; it does not support the
newer `--clearenv` flag. The runtime contains no host `/usr`, model weights or
credentials. Its `/tmp` aliases the sample's `scratch/tmp` and shares its byte
accounting. No host proc or GPU device is mounted.

On shared filesystems, verify actual execution, file locking and publication on
the target node. If direct execution is unsuitable, keep the bundle on persistent
storage and prepare a node-local cache once, not once per query.

The current `fuse.quarkfs_client` AFS mount passed single import and execution,
but concurrent `flock` failed with `EAGAIN` (and explicit unlock returned
`ENOENT`). It is therefore **not an accepted shared cache location on this
node**. Keep the bundle on AFS and use the configured local cache. If `/tmp` is
cleared during container restart, import the same bundle once again; this needs
no registry connection. The manager reports lock failures and does not silently
switch cache locations or repair published entries.

## Run through the current task or v2 entrypoint

Add `--runtime-config` to `scripts/run_task.py`, `scripts/run_document_pool.py`,
or `scripts/run_benchmark_v2.py`. Use a new output directory for Frozen runs.
For example, with an already running model service:

```bash
python scripts/run_benchmark_v2.py --suite /path/to/frozen-suite --split smoke \
  --backend-manifest /path/to/backend.json \
  --runtime-config configs/runtime/frozen-debian13.json \
  --output /path/to/new-frozen-smoke --execute
```

The four-5090 launcher accepts the configuration through `DCI_RUNTIME_CONFIG` and
an optional cold-cache bundle through `DCI_RUNTIME_BUNDLE`:

```bash
DCI_RUNTIME_CONFIG="$PWD/configs/runtime/frozen-debian13.json" \
DCI_RUNTIME_BUNDLE="$PWD/results/frozen-runtime/20261111-xaDmpt/release-v2/rootfs.tar" \
  bash scripts/launch_benchmark20_qwen38_5090.sh --preflight-only
```

The launcher checks or prepares host `/usr/bin/bwrap` first (Frozen mode does not
require host ripgrep), then verifies the runtime and exercises the actual sandbox
and synthetic Inspect bridge before model loading. Its default dry-run does not
install dependencies or unpack a bundle and does not start a model.
The historical `run_pool_smoke.sh` re-executes an immutable source bundle and
explicitly rejects Frozen settings; use the current task/v2 entrypoints instead.
Do not patch historical sealed bundles to add this feature.

`run_benchmark_v2.py --preflight-only` remains a data preflight. It does not replace
`check_frozen_runtime.py` or the synthetic bridge acceptance. Existing `read`
receipt rules still apply: Bash reading a document does not create a receipt.

## Open Bash prompt revision (2026-11-13)

The current shared Document prompts let the model choose its retrieval strategy
with any executable already present in the runtime, shell builtins, pipelines,
loops and scripts. They prescribe no command or search order. Frozen mode exposes
the shipped Debian userland, not every package available for Debian; Python is
not added and evaluation cannot install packages or access the network.
Intermediate files belong in `/scratch`, temporary files in `$TMPDIR`. Corpus,
tool output, process and token limits still apply. The stable-rg note describes
optional rg behavior, not a preferred retrieval strategy.

The active Code Track prompt also drops its rg-first wording while retaining
its prohibition on editing corpus files or executing repository code. Its
runtime stays host-tools; this prompt revision does not enable Frozen for Code.

At least one nonempty `read` receipt remains mandatory; a separate receipt for
every submitted ID is not required. If a forced read succeeds with budget and
tool/turn capacity left, the existing controller can reopen both tools. Final
and final-repair phases keep tools disabled. V2 also enters that final-repair
state after an autonomous malformed final answer with existing read evidence;
invalid JSON no longer reopens retrieval through the generic repair prompt.

The prompt revision changes the protocol hash and source fingerprint, not the
RootFS content or runtime policy identity. Historical prompts and results remain
in their sealed artifacts. Host/Frozen comparisons require both arms to use the
same new prompts; comparing new Frozen results with the old rg-first Host smoke
does not isolate the runtime effect. See the
[Qwen3.8-27B 40-execution smoke plan](frozen_qwen38_smoke_20261113.md).

## Identity and recovery

Runtime content identity consists of the final image manifest digest, platform
and normalized rootfs tree hash. Execution policy has a separate hash covering
mounts, environment, resource limits and the stable-rg configuration. Absolute
cache and bundle paths do not belong to logical identity.

Python verifies the runtime before sample execution and passes a verified
descriptor to Pi. Pi checks the READY identity and executable hashes, and emits
`agent_runtime` in its final record and trace. The parent rejects mismatches.
The same identity is carried in the run manifest's `code.agent_runtime`, sample
provenance, v2 launch and summary. Resume, completed-batch reuse and development
calibration/selection must agree on it. Historical records without this identity
cannot be reused as Frozen results.

The kernel, host bwrap, Node and Inspect are still execution dependencies. A
frozen rootfs does not guarantee identical latency or model scores. Keep hardware
and elapsed time descriptive; the existing per-Bash limits remain in force.

## Build and publish a new bundle

`scripts/build_frozen_runtime.py` is a separate, explicit build-time command. It
uses a verified crane binary, a pinned Debian OCI digest and signed APT packages
from a dated snapshot. It requires network access only during preparation and
does not require Docker or Podman. Verify crane's release archive checksum before
using it, then supply the executable's SHA-256:

```bash
python scripts/build_frozen_runtime.py \
  --crane /path/to/verified/crane --crane-sha256 EXECUTABLE_SHA256 \
  --output /path/to/new-release
```

An optional `--prepared-rootfs` copies an explicitly trusted build tree and skips
acquisition. The release records that source kind; it is not proof that arbitrary
supplied files originated from the declared base image.

Debian's absolute alternatives symlinks are normalized to equivalent relative
links at the build boundary. The builder freezes permissions and produces a new
single-layer OCI archive matching those bytes. Its **new image digest**, not the
original Debian digest, identifies the runtime. Output includes `rootfs.tar`,
`runtime-image.tar`, `runtime-config.json`, `release.json`, `packages.txt`, and
`tool-inventory.json`. The localhost image reference is an offline identity, not
a claim of registry publication. No push is performed.

The optional `docker/runtime/Dockerfile` offers the corresponding CI package
build. The complete shipped image includes the Debian base tools and dependencies;
the short package list is not a command-name allowlist. Python is not added in
the first runtime. Any tool-set update gets new content identity and separate
evaluation/calibration results.

Validate a new release with cache tests, real Pi sandbox tests, and the synthetic
Inspect bridge before a real-model smoke. A model smoke and development-set
ablation are separate acceptance stages; no formal ranking is inferred from
synthetic tests. Retain prior releases and result directories.

## Validated local release (2026-11-11)

The current development release is stored at
`results/frozen-runtime/20261111-xaDmpt/release-v2/`. It is an offline artifact,
not a registry publication. Its image digest is
`sha256:257287ccc124681dd7fad47c24685f507332592b4b94ed241f7e699853d62f58`.
The checked configuration pins its bundle and tree digests. The release records
the exact package/tool inventories and builder hash. It was assembled through
`--prepared-rootfs` after the pinned-base/signed-snapshot preparation; the
builder's full default acquisition path and optional Docker build have not been
accepted end to end.

From this checkout, these commands use the existing artifact without network
access or a model:

```bash
DCI_PY=/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python
"$DCI_PY" -m dci_bench.runtime prepare \
  --config configs/runtime/frozen-debian13.json \
  --bundle results/frozen-runtime/20261111-xaDmpt/release-v2/rootfs.tar \
  --offline --preflight
"$DCI_PY" scripts/check_frozen_runtime.py \
  --runtime-config configs/runtime/frozen-debian13.json
```

To enable the real Frozen sandbox regressions as well as the ordinary Pi tests:

```bash
DCI_TEST_FROZEN_RUNTIME_JSON="$("$DCI_PY" -m dci_bench.runtime verify \
  --config configs/runtime/frozen-debian13.json)" npm --prefix pi-dci run test:dci
```

The environment value is the verified JSON descriptor, not a flag or a path.
Without it, tests requiring the real prepared RootFS are explicitly skipped.

The release passed real Pi execution and the synthetic Inspect bridge at seven
parallel processes followed by eight concurrent samples (15 samples total).
The bridge uses the generic document profile and synthetic documents; this is
not a pooled-data or model-quality benchmark. Final evidence, failure history,
and remaining acceptance stages are in §16 of
[the implementation plan](../DCI-Bench_Frozen_RootFS_Runtime_Design_and_Implementation.md).
The current container does not mount the launcher's default benchmark-v2 suite.
Real-model smoke, equal-budget Host/Frozen ablation, container-restart acceptance,
and promotion to the official default remain outstanding. Host tools remain
the default; Code Track is unchanged and rejects a Frozen descriptor.
