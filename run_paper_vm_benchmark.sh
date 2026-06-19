#!/usr/bin/env bash
set -Eeuo pipefail

# Reproduce the paper-style InferEdge benchmark on an Ubuntu VM and pull the
# results back to this WSL/local checkout.
#
# Default experiment matrix:
#   5 selected models x 3 representative inputs x 4 deployment mechanisms x 5 trials
#
# Typical use:
#   TARGET_PASSWORD='...' ./run_paper_vm_benchmark.sh
#   ./run_paper_vm_benchmark.sh status
#   ./run_paper_vm_benchmark.sh pull

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-all}"

TARGET_USER="${TARGET_USER:-zhang}"
TARGET_HOST="${TARGET_HOST:-192.168.43.200}"
TARGET_PORT="${TARGET_PORT:-2222}"
TARGET_PASSWORD="${TARGET_PASSWORD:-09231234}"
REMOTE_SUITE="${REMOTE_SUITE:-/home/${TARGET_USER}/Desktop/CS4099Suite}"

TRIALS="${TRIALS:-5}"
SET_NAME="${SET_NAME:-paper_vm_selected_5models_3inputs_5trials_$(date +%Y%m%d_%H%M%S)}"
MECHANISMS="${MECHANISMS:-native,docker,wasm_interpreted,wasm_jit,wasm_aot}"
ANALYZED_RESULTS_DIR="${ANALYZED_RESULTS_DIR:-analyzed_results_paper}"
DOCKER_OVERHEAD_VIEW="${DOCKER_OVERHEAD_VIEW:-2}"
SIGNIFICANCE_LEVEL="${SIGNIFICANCE_LEVEL:-0.05}"
REQUIRE_CGROUP_VERSION="${REQUIRE_CGROUP_VERSION:-v1}"

ALLOW_MISSING_METRICS="${ALLOW_MISSING_METRICS:-1}"
SKIP_PERF="${SKIP_PERF:-0}"
RUN_SETUP="${RUN_SETUP:-0}"
OVERWRITE_REMOTE_RESULTS="${OVERWRITE_REMOTE_RESULTS:-0}"
OVERWRITE_LOCAL_RESULTS="${OVERWRITE_LOCAL_RESULTS:-0}"
OVERWRITE_ANALYSIS="${OVERWRITE_ANALYSIS:-1}"
RESUME_COMPLETED="${RESUME_COMPLETED:-1}"
SKIP_MODEL_GENERATION="${SKIP_MODEL_GENERATION:-0}"
SYNC_BASE_ASSETS="${SYNC_BASE_ASSETS:-1}"

MODELS=(
  efficientnet_b3.pt
  efficientnet_b5.pt
  mobilenetv3_small.pt
  resnet18.pt
  resnet50.pt
)

INPUTS=(
  CIFAR10_00013.png
  CIFAR100_00788.png
  ILSVRC2012_test_00000036.JPEG
)

MODEL_LIST="${MODEL_LIST:-${MODELS[*]}}"
INPUT_LIST="${INPUT_LIST:-${INPUTS[*]}}"

usage() {
  cat <<EOF
Usage:
  ./run_paper_vm_benchmark.sh             # generate/check assets, sync VM, run, pull, analyze
  ./run_paper_vm_benchmark.sh all         # same as above
  ./run_paper_vm_benchmark.sh sync        # only sync required assets to VM
  ./run_paper_vm_benchmark.sh run         # run benchmark on VM in the foreground
  ./run_paper_vm_benchmark.sh background  # start benchmark on VM with nohup
  ./run_paper_vm_benchmark.sh status      # show remote PID/log tail and CSV count
  ./run_paper_vm_benchmark.sh pull        # pull remote result set and run local analysis
  ./run_paper_vm_benchmark.sh analyze     # analyze an already pulled local result set

Important environment variables:
  TARGET_USER=$TARGET_USER
  TARGET_HOST=$TARGET_HOST
  TARGET_PORT=$TARGET_PORT
  TARGET_PASSWORD=<optional, used by sshpass if password SSH is needed>
  REMOTE_SUITE=$REMOTE_SUITE
  SET_NAME=$SET_NAME
  TRIALS=$TRIALS
  MECHANISMS=$MECHANISMS
  REQUIRE_CGROUP_VERSION=$REQUIRE_CGROUP_VERSION
  RESUME_COMPLETED=$RESUME_COMPLETED

Examples:
  TARGET_PASSWORD='your-ssh-password' ./run_paper_vm_benchmark.sh
  TRIALS=5 TARGET_PASSWORD='your-ssh-password' ./run_paper_vm_benchmark.sh
  SET_NAME=paper_vm_selected_5models_3inputs_5trials_20260527 ./run_paper_vm_benchmark.sh status
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

python_bin() {
  if [[ -x "$ROOT_DIR/myenv/bin/python" ]]; then
    printf '%s\n' "$ROOT_DIR/myenv/bin/python"
  else
    command -v python3
  fi
}

remote_target() {
  printf '%s@%s' "$TARGET_USER" "$TARGET_HOST"
}

ssh_prefix() {
  if [[ -n "$TARGET_PASSWORD" ]]; then
    need_cmd sshpass
    SSHPASS="$TARGET_PASSWORD" sshpass -e "$@"
  else
    "$@"
  fi
}

ssh_cmd() {
  ssh_prefix ssh \
    -p "$TARGET_PORT" \
    -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=6 \
    "$(remote_target)" "$@"
}

ssh_tty_cmd() {
  ssh_prefix ssh -tt \
    -p "$TARGET_PORT" \
    -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=6 \
    "$(remote_target)" "$@"
}

scp_cmd() {
  ssh_prefix scp \
    -P "$TARGET_PORT" \
    -o StrictHostKeyChecking=accept-new \
    "$@"
}

remote_quote() {
  printf '%q' "$1"
}

ensure_local_models() {
  local py
  py="$(python_bin)"
  local model_dir="$ROOT_DIR/models/models"
  mkdir -p "$model_dir"

  local missing=()
  for model in $MODEL_LIST; do
    if [[ ! -f "$model_dir/$model" ]]; then
      missing+=("$model")
    fi
  done

  if (( ${#missing[@]} == 0 )); then
    log "All required local model files are present."
    return
  fi

  if [[ "$SKIP_MODEL_GENERATION" == "1" ]]; then
    printf 'Missing models:\n' >&2
    printf '  %s\n' "${missing[@]}" >&2
    die "Model generation is disabled by SKIP_MODEL_GENERATION=1."
  fi

  log "Generating missing TorchScript models: ${missing[*]}"

  local eff_flags=()
  local mob_flags=()
  local res_flags=()
  for model in "${missing[@]}"; do
    case "$model" in
      efficientnet_b0.pt) eff_flags+=(--b0) ;;
      efficientnet_b1.pt) eff_flags+=(--b1) ;;
      efficientnet_b2.pt) eff_flags+=(--b2) ;;
      efficientnet_b3.pt) eff_flags+=(--b3) ;;
      efficientnet_b4.pt) eff_flags+=(--b4) ;;
      efficientnet_b5.pt) eff_flags+=(--b5) ;;
      mobilenetv3_small.pt) mob_flags+=(--mobilenetv3_small) ;;
      mobilenetv3_large.pt) mob_flags+=(--mobilenetv3_large) ;;
      resnet18.pt) res_flags+=(--resnet18) ;;
      resnet34.pt) res_flags+=(--resnet34) ;;
      resnet50.pt) res_flags+=(--resnet50) ;;
      *) die "No generation rule for model: $model" ;;
    esac
  done

  if (( ${#eff_flags[@]} > 0 )); then
    (cd "$model_dir" && "$py" "$ROOT_DIR/host_scripts/model_generation/gen_efficientnet_models.py" "${eff_flags[@]}")
  fi
  if (( ${#mob_flags[@]} > 0 )); then
    (cd "$model_dir" && "$py" "$ROOT_DIR/host_scripts/model_generation/gen_mobilenet_models.py" "${mob_flags[@]}")
  fi
  if (( ${#res_flags[@]} > 0 )); then
    (cd "$model_dir" && "$py" "$ROOT_DIR/host_scripts/model_generation/gen_resnet_models.py" "${res_flags[@]}")
  fi

  for model in $MODEL_LIST; do
    [[ -f "$model_dir/$model" ]] || die "Model still missing after generation: $model"
  done
}

ensure_local_inputs() {
  local input_dir="$ROOT_DIR/inputs/inputs"
  for input in $INPUT_LIST; do
    [[ -f "$input_dir/$input" ]] || die "Required representative input not found: $input_dir/$input"
  done
  log "All required representative input files are present."
}

sync_base_assets() {
  if [[ "$SYNC_BASE_ASSETS" != "1" ]]; then
    log "Skipping base asset sync because SYNC_BASE_ASSETS=$SYNC_BASE_ASSETS."
    return 0
  fi

  log "Syncing suite runtime assets to $(remote_target):$REMOTE_SUITE"
  ssh_cmd "mkdir -p $(remote_quote "$REMOTE_SUITE") $(remote_quote "$REMOTE_SUITE/models") $(remote_quote "$REMOTE_SUITE/inputs") $(remote_quote "$REMOTE_SUITE/logs") $(remote_quote "$REMOTE_SUITE/results")"

  local dirs=(native wasm libtorch cadvisor prometheus docker python target_scripts)
  for dir in "${dirs[@]}"; do
    [[ -e "$ROOT_DIR/$dir" ]] || die "Required local asset missing: $ROOT_DIR/$dir"
    scp_cmd -r "$ROOT_DIR/$dir" "$(remote_target):$REMOTE_SUITE/"
  done

  scp_cmd "$ROOT_DIR/data_scripts/collect_data.py" "$(remote_target):$REMOTE_SUITE/collect_data.py"

  if [[ -d "$ROOT_DIR/wasmedge/bin" ]]; then
    log "Syncing WasmEdge runtime to remote ~/.wasmedge"
    ssh_cmd "mkdir -p /home/$(remote_quote "$TARGET_USER")/.wasmedge"
    scp_cmd -r "$ROOT_DIR/wasmedge/." "$(remote_target):/home/$TARGET_USER/.wasmedge/"
  fi
}

sync_selected_models_and_inputs() {
  local model_dir="$ROOT_DIR/models/models"
  local input_dir="$ROOT_DIR/inputs/inputs"

  log "Syncing selected models."
  for model in $MODEL_LIST; do
    scp_cmd "$model_dir/$model" "$(remote_target):$REMOTE_SUITE/models/"
  done

  log "Syncing selected representative inputs."
  for input in $INPUT_LIST; do
    scp_cmd "$input_dir/$input" "$(remote_target):$REMOTE_SUITE/inputs/"
  done
}

write_remote_runner() {
  local tmp_script
  tmp_script="$(mktemp "${TMPDIR:-/tmp}/inferedge-remote-runner.XXXXXX.sh")"
  cat > "$tmp_script" <<'REMOTE_SCRIPT'
#!/usr/bin/env bash
set -Eeuo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

TRIALS="${TRIALS:?TRIALS is required}"
SET_NAME="${SET_NAME:?SET_NAME is required}"
MECHANISMS="${MECHANISMS:?MECHANISMS is required}"
MODEL_LIST="${MODEL_LIST:?MODEL_LIST is required}"
INPUT_LIST="${INPUT_LIST:?INPUT_LIST is required}"
ALLOW_MISSING_METRICS="${ALLOW_MISSING_METRICS:-1}"
SKIP_PERF="${SKIP_PERF:-0}"
OVERWRITE_REMOTE_RESULTS="${OVERWRITE_REMOTE_RESULTS:-0}"
RUN_SETUP="${RUN_SETUP:-0}"
REQUIRE_CGROUP_VERSION="${REQUIRE_CGROUP_VERSION:-v1}"
RESUME_COMPLETED="${RESUME_COMPLETED:-1}"

cd "$(dirname "${BASH_SOURCE[0]}")"
SUITE_DIR="$(pwd)"
export USERNAME="$(whoami)"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.wasmedge/lib64:$SUITE_DIR/libtorch/lib"
export PATH="$PATH:$HOME/.wasmedge/bin"

if [[ -f /sys/fs/cgroup/cgroup.controllers ]]; then
  CGROUP_MODE="v2"
else
  CGROUP_MODE="v1_or_hybrid"
fi

case "$REQUIRE_CGROUP_VERSION" in
  v1)
    if [[ "$CGROUP_MODE" == "v2" ]]; then
      die "This benchmark is forced to cgroup v1, but the Ubuntu VM is running cgroup v2. Reboot the VM with systemd.unified_cgroup_hierarchy=0 and retry."
    fi
    ;;
  v2)
    if [[ "$CGROUP_MODE" != "v2" ]]; then
      die "This benchmark is forced to cgroup v2, but the Ubuntu VM is running $CGROUP_MODE."
    fi
    ;;
  any)
    ;;
  *)
    die "Invalid REQUIRE_CGROUP_VERSION=$REQUIRE_CGROUP_VERSION. Use v1, v2, or any."
    ;;
esac

mkdir -p logs results
LOG_FILE="logs/${SET_NAME}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ -d "results/$SET_NAME" && "$OVERWRITE_REMOTE_RESULTS" != "1" && "$RESUME_COMPLETED" != "1" ]]; then
  if find "results/$SET_NAME" -maxdepth 1 -type f | grep -q .; then
    die "results/$SET_NAME already exists on the VM. Set RESUME_COMPLETED=1, OVERWRITE_REMOTE_RESULTS=1, or choose a new SET_NAME."
  fi
fi

if [[ "$OVERWRITE_REMOTE_RESULTS" == "1" ]]; then
  rm -rf "results/$SET_NAME"
fi
mkdir -p "results/$SET_NAME"

if [[ "$RUN_SETUP" == "1" ]]; then
  log "Running one-time target setup. This may ask for sudo."
  sudo "$SUITE_DIR/target_scripts/setup.sh"
fi

if [[ ! -x myenv/bin/python ]]; then
  log "Creating target Python virtualenv."
  python3 -m venv myenv
  myenv/bin/pip install -r python/target/requirements.txt
fi

if ! myenv/bin/python - <<'PY' >/dev/null 2>&1
import requests
PY
then
  log "Installing target Python requirements."
  myenv/bin/pip install -r python/target/requirements.txt
fi

chmod +x native/torch_image_classification 2>/dev/null || true
chmod +x cadvisor/cadvisor 2>/dev/null || true
chmod +x prometheus/prometheus 2>/dev/null || true
chmod +x "$HOME/.wasmedge/bin/wasmedge" 2>/dev/null || true

if [[ ! -x "$HOME/.wasmedge/bin/wasmedge" ]]; then
  die "WasmEdge binary not found at $HOME/.wasmedge/bin/wasmedge"
fi

if [[ ",$MECHANISMS," == *",wasm_aot,"* ]]; then
  log "Compiling wasm/aot.wasm with WasmEdge O0 AOT."
  "$HOME/.wasmedge/bin/wasmedge" compile --optimize 0 wasm/interpreted.wasm wasm/aot.wasm
fi

if [[ "$(uname -m)" == "aarch64" ]]; then
  ARCH="arm64"
else
  ARCH="amd64"
fi

if [[ ",$MECHANISMS," == *",docker,"* ]]; then
  log "Checking Docker image image-classification:$ARCH"
  if ! sudo docker image inspect "image-classification:$ARCH" >/dev/null 2>&1; then
    sudo docker load -i "docker/image-classification-${ARCH}.tar"
  fi
fi

for model in $MODEL_LIST; do
  [[ -f "models/$model" ]] || die "Missing remote model: models/$model"
done

for input in $INPUT_LIST; do
  [[ -f "inputs/$input" ]] || die "Missing remote input: inputs/$input"
done

{
  echo "set_name=$SET_NAME"
  echo "started_at=$(date -Is)"
  echo "suite_dir=$SUITE_DIR"
  echo "platform=Ubuntu VM"
  echo "uname=$(uname -a)"
  echo "arch=$ARCH"
  echo "trials=$TRIALS"
  echo "mechanisms=$MECHANISMS"
  echo "models=$MODEL_LIST"
  echo "inputs=$INPUT_LIST"
  echo "cgroup=$CGROUP_MODE"
  echo "required_cgroup=$REQUIRE_CGROUP_VERSION"
  echo "resume_completed=$RESUME_COMPLETED"
} | tee "results/$SET_NAME/manifest.txt"

COLLECT_OPTIONS=()
if [[ "$ALLOW_MISSING_METRICS" == "1" ]]; then
  COLLECT_OPTIONS+=(--allow_missing_metrics)
fi
if [[ "$SKIP_PERF" == "1" ]]; then
  COLLECT_OPTIONS+=(--skip-perf)
fi

log "Starting benchmark: $SET_NAME"
log "Matrix: $(wc -w <<< "$MODEL_LIST") models x $(wc -w <<< "$INPUT_LIST") inputs x $MECHANISMS x $TRIALS trials"
log "Resume completed model/input pairs: $RESUME_COMPLETED"

csv_has_data_rows() {
  local csv_file="$1"
  [[ -f "$csv_file" ]] || return 1
  [[ "$(wc -l < "$csv_file")" -gt 1 ]]
}

result_pair_complete() {
  local model="$1"
  local input="$2"
  local prefix="results/$SET_NAME/${model}&${input}"
  csv_has_data_rows "${prefix}&time_results.csv" || return 1
  if [[ "$SKIP_PERF" != "1" ]]; then
    csv_has_data_rows "${prefix}&perf_results.csv" || return 1
  fi
  return 0
}

completed_pairs=0
skipped_pairs=0

for model in $MODEL_LIST; do
  for input in $INPUT_LIST; do
    if [[ "$RESUME_COMPLETED" == "1" ]] && result_pair_complete "$model" "$input"; then
      log "Skipping completed model=$model input=$input"
      skipped_pairs=$((skipped_pairs + 1))
      continue
    fi

    log "Running model=$model input=$input"
    myenv/bin/python collect_data.py \
      --model "$model" \
      --input "$input" \
      --trials "$TRIALS" \
      --set_name "$SET_NAME" \
      --mechanisms "$MECHANISMS" \
      --arch "$ARCH" \
      "${COLLECT_OPTIONS[@]}"
    completed_pairs=$((completed_pairs + 1))
  done
done

log "Benchmark complete: $SET_NAME"
log "Pairs run this time: $completed_pairs"
log "Pairs skipped as complete: $skipped_pairs"
log "CSV files written: $(find "results/$SET_NAME" -maxdepth 1 -type f -name '*_results.csv' | wc -l)"
REMOTE_SCRIPT

  scp_cmd "$tmp_script" "$(remote_target):$REMOTE_SUITE/run_paper_benchmark_remote.sh"
  rm -f "$tmp_script"
  ssh_cmd "chmod +x $(remote_quote "$REMOTE_SUITE/run_paper_benchmark_remote.sh")"
}

sync_all() {
  ensure_local_models
  ensure_local_inputs
  sync_base_assets
  sync_selected_models_and_inputs
  write_remote_runner
  log "Sync complete."
}

remote_env_prefix() {
  printf 'TRIALS=%q SET_NAME=%q MECHANISMS=%q MODEL_LIST=%q INPUT_LIST=%q ALLOW_MISSING_METRICS=%q SKIP_PERF=%q OVERWRITE_REMOTE_RESULTS=%q RUN_SETUP=%q RESUME_COMPLETED=%q' \
    "$TRIALS" "$SET_NAME" "$MECHANISMS" "$MODEL_LIST" "$INPUT_LIST" \
    "$ALLOW_MISSING_METRICS" "$SKIP_PERF" "$OVERWRITE_REMOTE_RESULTS" "$RUN_SETUP" "$RESUME_COMPLETED"
  printf ' REQUIRE_CGROUP_VERSION=%q' "$REQUIRE_CGROUP_VERSION"
}

run_remote_foreground() {
  log "Running benchmark on VM in foreground. Remote sudo may ask for a password."
  ssh_tty_cmd "cd $(remote_quote "$REMOTE_SUITE") && $(remote_env_prefix) bash ./run_paper_benchmark_remote.sh"
}

run_remote_background() {
  log "Starting benchmark on VM with nohup."
  ssh_cmd "cd $(remote_quote "$REMOTE_SUITE") && mkdir -p logs && { nohup env $(remote_env_prefix) bash ./run_paper_benchmark_remote.sh > logs/$(remote_quote "$SET_NAME").nohup.out 2>&1 & echo \$! > logs/$(remote_quote "$SET_NAME").pid; cat logs/$(remote_quote "$SET_NAME").pid; }"
  log "Started. Use SET_NAME=$SET_NAME ./run_paper_vm_benchmark.sh status"
  log "When it finishes, use SET_NAME=$SET_NAME ./run_paper_vm_benchmark.sh pull"
}

status_remote() {
  ssh_cmd "cd $(remote_quote "$REMOTE_SUITE") && echo 'set=$SET_NAME' && if [[ -f logs/$(remote_quote "$SET_NAME").pid ]]; then echo -n 'pid='; cat logs/$(remote_quote "$SET_NAME").pid; fi && if [[ -d results/$(remote_quote "$SET_NAME") ]]; then echo -n 'csv_count='; find results/$(remote_quote "$SET_NAME") -maxdepth 1 -type f -name '*_results.csv' | wc -l; fi && if [[ -f logs/$(remote_quote "$SET_NAME").log ]]; then echo '--- log tail ---'; tail -n 40 logs/$(remote_quote "$SET_NAME").log; fi"
}

pull_results() {
  local local_results="$ROOT_DIR/results/$SET_NAME"
  if [[ -e "$local_results" && "$OVERWRITE_LOCAL_RESULTS" != "1" ]]; then
    die "Local result directory already exists: $local_results. Set OVERWRITE_LOCAL_RESULTS=1 or choose SET_NAME explicitly."
  fi
  if [[ -e "$local_results" ]]; then
    rm -rf "$local_results"
  fi

  mkdir -p "$ROOT_DIR/results"
  log "Pulling VM results to $ROOT_DIR/results/$SET_NAME"
  scp_cmd -r "$(remote_target):$REMOTE_SUITE/results/$SET_NAME" "$ROOT_DIR/results/"
  scp_cmd "$(remote_target):$REMOTE_SUITE/logs/$SET_NAME.log" "$local_results/full_benchmark_${SET_NAME}.log" || true
}

csv_metrics_from_header() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  head -n 1 "$file" | tr -d '\r' | cut -d ',' -f 4-
}

combine_metrics() {
  local time_file="$1"
  local perf_file="$2"
  local time_metrics perf_metrics
  time_metrics="$(csv_metrics_from_header "$time_file")"
  perf_metrics="$(csv_metrics_from_header "$perf_file")"

  if [[ -n "$time_metrics" && -n "$perf_metrics" ]]; then
    printf '%s,%s\n' "$time_metrics" "$perf_metrics"
  elif [[ -n "$time_metrics" ]]; then
    printf '%s\n' "$time_metrics"
  else
    printf '%s\n' "$perf_metrics"
  fi
}

analyze_results() {
  local py
  py="$(python_bin)"
  local set_dir="$ROOT_DIR/results/$SET_NAME"
  [[ -d "$set_dir" ]] || die "Local result set not found: $set_dir"

  local analyzed_dir="$set_dir/$ANALYZED_RESULTS_DIR"
  if [[ -e "$analyzed_dir" && "$OVERWRITE_ANALYSIS" == "1" ]]; then
    rm -rf "$analyzed_dir"
  fi
  mkdir -p "$analyzed_dir"

  shopt -s nullglob
  local time_files=("$set_dir"/*'&time_results.csv')
  shopt -u nullglob
  (( ${#time_files[@]} > 0 )) || die "No time result CSV files found in $set_dir"

  log "Analyzing ${#time_files[@]} model/input experiments into $analyzed_dir"
  for time_file in "${time_files[@]}"; do
    local base model rest input perf_file metrics
    base="$(basename "$time_file")"
    model="${base%%&*}"
    rest="${base#*&}"
    input="${rest%&time_results.csv}"
    perf_file="$set_dir/${model}&${input}&perf_results.csv"
    metrics="$(combine_metrics "$time_file" "$perf_file")"

    "$py" "$ROOT_DIR/data_scripts/analyze_data.py" \
      --experiment-set "$SET_NAME" \
      --model "$model" \
      --input "$input" \
      --significance-level "$SIGNIFICANCE_LEVEL" \
      --docker-overhead-view "$DOCKER_OVERHEAD_VIEW" \
      --mechanisms "$MECHANISMS" \
      --metrics "$metrics" \
      --analyzed-results-dir "$ANALYZED_RESULTS_DIR"
  done

  log "Analysis complete: $analyzed_dir/aggregate_results.csv"
}

print_outputs() {
  log "Done."
  log "Local raw results: $ROOT_DIR/results/$SET_NAME"
  log "Local aggregate CSV: $ROOT_DIR/results/$SET_NAME/$ANALYZED_RESULTS_DIR/aggregate_results.csv"
  log "Remote raw results kept at: $REMOTE_SUITE/results/$SET_NAME"
}

main() {
  need_cmd ssh
  need_cmd scp
  case "$MODE" in
    all|paper|one-shot|oneshot)
      sync_all
      run_remote_foreground
      pull_results
      analyze_results
      print_outputs
      ;;
    sync)
      sync_all
      ;;
    run)
      run_remote_foreground
      ;;
    background)
      sync_all
      run_remote_background
      ;;
    status)
      status_remote
      ;;
    pull)
      pull_results
      analyze_results
      ;;
    analyze)
      analyze_results
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      usage
      die "Unknown mode: $MODE"
      ;;
  esac
}

main "$@"
