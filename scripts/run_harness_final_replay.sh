#!/usr/bin/env bash
# Called only by the frozen bundle after all CPU integrity checks pass.
set -Eeuo pipefail
bundle="$1"
output="$2"
runtime="$3"
cd "$runtime"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$(dirname "$SERVING_PYTHON"):$PATH"
SERVICE_PID=""
cleanup() {
  if [[ -n "$SERVICE_PID" ]]; then
    kill -TERM -- "-$SERVICE_PID" 2>/dev/null || true
    wait "$SERVICE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for command in curl setsid; do command -v "$command" >/dev/null; done
# Replay instantiates no search/read tools. It needs neither bubblewrap nor rg.
# Reject an occupied port before loading weights, including non-HTTP listeners.
"$BENCH_PYTHON" -c 'import os,socket; s=socket.socket(); s.bind(("127.0.0.1", int(os.environ["VLLM_PORT"]))); s.close()'
"$SERVING_PYTHON" scripts/harness_pilot_runtime.py --output "$output/runtime.json"
"$BENCH_PYTHON" -c 'import json,sys; a=json.load(open(sys.argv[1]))["backend"]["runtime"]; b=json.load(open(sys.argv[2])); assert a["packages"]==b["packages"], "serving packages changed"; assert a["torch_cuda"]==b["torch_cuda"], "CUDA runtime changed"; assert [(g["name"],g["capability"]) for g in a["cuda"]["gpus"]]==[(g["name"],g["capability"]) for g in b["cuda"]["gpus"]], "GPU configuration changed"' "$bundle/expected-backend.json" "$output/runtime.json"
setsid bash scripts/serve_vllm_qwen38.sh >"$output/server.log" 2>&1 &
SERVICE_PID=$!
ready=0
for ((attempt=0; attempt<360; attempt+=1)); do
  if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
    echo "[replay] serving failed; inspect $output/server.log" >&2
    exit 2
  fi
  if curl --noproxy '*' -fsS "$BACKEND_BASE_URL/models" >"$output/models.json" 2>/dev/null; then ready=1; break; fi
  sleep 5
done
[[ "$ready" == "1" ]] || { echo "[replay] serving readiness timed out" >&2; exit 2; }
VLLM_VERSION="$("$SERVING_PYTHON" -c 'from importlib.metadata import version; print(version("vllm"))')"
"$BENCH_PYTHON" -m dci_bench.backends.manifest \
  --output "$output/backend.json" --model-key "$SERVED_MODEL_NAME" \
  --model-path "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME" --base-url "$BACKEND_BASE_URL" \
  --vllm-version "$VLLM_VERSION" --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --language-model-only --sampling-backend auto --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --context-length 65536 --dtype "$DTYPE" --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" --runtime-info "$output/runtime.json"
"$BENCH_PYTHON" -c 'import json,sys; a=json.load(open(sys.argv[1]))["model"]; b=json.load(open(sys.argv[2]))["model"]; assert a==b, "model configuration or weight provenance changed"' "$bundle/expected-backend.json" "$output/backend.json"
PATH="$(dirname "$BENCH_PYTHON"):$PATH" "$BENCH_PYTHON" scripts/harness_pilot_warmup.py \
  --base-url "$BACKEND_BASE_URL" --model "$SERVED_MODEL_NAME" --output "$output/warmup.json" \
  --concurrency 4 --deadline-seconds 600 --http-timeout-seconds 600
PATH="$(dirname "$BENCH_PYTHON"):$PATH" "$BENCH_PYTHON" "$bundle/replay.py" \
  --bundle "$bundle" --evaluate --output "$output"
