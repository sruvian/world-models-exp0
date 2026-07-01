#!/bin/bash
YAML_DIR="seed_configs/"
LOG_DIR="logs/"
mkdir -p "$LOG_DIR"

HEADLINE_LATENT=16
LATENTS=(3 8 16 32 64)
SEED_VALUES=(0 1 2 3 4)

run_batch() {
    local yamls=("$@")
    local pids=()
    for yaml in "${yamls[@]}"; do
        name=$(basename "$yaml" .yaml)
        OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        python main.py --yaml "$yaml" > "$LOG_DIR/${name}.log" 2>&1 &
        pids+=($!)
        echo "[START] $name (PID $!)"
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || echo "[FAILED] PID $pid" >> "$LOG_DIR/failures.log"
    done
    echo "[BATCH DONE]"
}

run_yaml_modifier() {
    local latent=$1
    local seed=$2
    python yaml_modifier.py \
        --yaml_dir "$YAML_DIR" \
        --latent "$latent" \
        --seed "$seed"
}

run_all_yamls() {
    local latent=$1
    local seed=$2
    echo "==============================="
    echo "Running latent=$latent seed=$seed"
    echo "==============================="
    run_yaml_modifier "$latent" "$seed"
    local yamls=($(ls "$YAML_DIR"/*.yaml))
    for ((i=0; i<${#yamls[@]}; i+=6)); do
        batch=("${yamls[@]:i:6}")
        echo "[BATCH $((i/6 + 1))] ${batch[@]}"
        run_batch "${batch[@]}"
    done
}

for seed in "${SEED_VALUES[@]}"; do
    run_all_yamls "$HEADLINE_LATENT" "$seed"
done


for latent in "${LATENTS[@]}"; do
    if [ "$latent" -eq "$HEADLINE_LATENT" ]; then
        continue
    fi
    run_all_yamls "$latent" 0
done

echo "All experiments complete"