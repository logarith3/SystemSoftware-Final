#!/usr/bin/env bash
set -euo pipefail

# Directory containing compiled binaries.
BIN_DIR="bin"
TRAINER="${BIN_DIR}/trainer"

# Directory for log and model files.
LOG_DIR="logs"

# Training and test datasets.
TRAIN_CSV="data/train.csv"
TEST_CSV="data/test.csv"

mkdir -p "$LOG_DIR"

# Build binaries if trainer is missing.
if [[ ! -x "$TRAINER" ]]; then
  echo "[run] Trainer binary not found, building..."
  ./build.sh
fi

# Location for model parameters (used by backward_layer).
export MODEL_FILE="${MODEL_FILE:-logs/model_params.txt}"

# --------------------------------------------------------------------
# progress_bar
# --------------------------------------------------------------------
progress_bar() {
  local current=$1
  local total=$2
  local width=40

  if [[ "$total" -le 0 ]]; then
    printf "\r[progress] processing..."
    return
  fi

  local percent=$(( 100 * current / total ))
  local filled=$(( width * current / total ))
  local empty=$(( width - filled ))

  printf "\r["
  for ((i=0; i<filled; i++)); do printf "#"; done
  for ((i=0; i<empty; i++)); do printf "."; done
  printf "] %3d%% (%d/%d)" "$percent" "$current" "$total"
}

# --------------------------------------------------------------------
# run_phase
# --------------------------------------------------------------------
run_phase() {
  local phase="$1"      # "train" or "test"
  local csv="$2"
  local log_file="$3"
  local err_file="$4"

  if [[ ! -f "$csv" ]]; then
    echo "[run] CSV file not found for phase '$phase': $csv" >&2
    return 1
  fi

  # 라인수 
  local total_lines
  total_lines=$(grep -cve '^\s*$' "$csv" || echo 0)

  echo "[run] Phase: $phase, file: $csv"
  echo "[run] Total lines: $total_lines"
  echo "[run] Logs: $log_file, errors: $err_file"

  local cnt=0

  # Pipeline:
  #   trainer stdout  -> tee (log_file + progress loop)
  #   trainer stderr  -> err_file
  BACKWARD_MODE="$phase" "$TRAINER" "$csv" 2> "$err_file" | \
  tee "$log_file" | \
  while IFS= read -r line; do
    if [[ "$line" == SAMPLE* ]]; then
      cnt=$((cnt + 1))
      progress_bar "$cnt" "$total_lines"
    fi
  done

  echo

  # ------------------------------------------------------------------
  # TODO: Extract and print phase summary from "$log_file".
  # ------------------------------------------------------------------
  
  local summary=""
  
  # grep으로 긁어오기
  summary=$(grep "SUMMARY" "$log_file" | tail -1)

  if [[ -n "$summary" ]]; then
    local t s l y
    
    # 읽기
    read -r t s l y <<< "$summary"

    echo "[run] Phase '$phase' summary: samples=$s avg_loss=$l avg_yhat=$y"
  else
    echo "[run] Phase '$phase' summary: (no SUMMARY line found)"
  fi

  echo "[run] Phase '$phase' finished."
  echo "[run] Final logs: $log_file"
}

# -------- PRE-TRAIN TEST (baseline) --------
run_phase "test" "$TEST_CSV" \
  "${LOG_DIR}/pre-test.log" \
  "${LOG_DIR}/pre-test.err"

# -------- TRAINING --------
run_phase "train" "$TRAIN_CSV" \
  "${LOG_DIR}/train-train.log" \
  "${LOG_DIR}/train-train.err"

# -------- POST-TRAIN TEST --------
run_phase "test" "$TEST_CSV" \
  "${LOG_DIR}/post-test.log" \
  "${LOG_DIR}/post-test.err"
