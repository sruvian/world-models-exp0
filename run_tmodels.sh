#!/bin/bash
set -u

YAML_DIR="combined_generated"
LOG_DIR="logs_seed"
mkdir -p "$LOG_DIR"

MAX_JOBS=$(( $(nproc) - 2 ))
PYTHON="$HOME/world-models-exp0/WSim/bin/python"
[ -x "$PYTHON" ] || { echo "no python at $PYTHON"; exit 1; }

mapfile -t yamls < <(find "$YAML_DIR" -name "*.yaml" | sort -r)
echo "Found ${#yamls[@]} YAMLs | $MAX_JOBS workers"

for yaml in "${yamls[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do wait -n; done
    name=$(basename "$yaml" .yaml)
    (
        echo "[START] $name"
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE "$PYTHON" main.py --yaml "$yaml" > "$LOG_DIR/${name}.log" 2>&1 \
            && echo "[DONE ] $name" \
            || echo "[FAIL ] $name" | tee -a "$LOG_DIR/failures.log"
    ) &
done
wait
echo "COMPLETE"