#!/bin/bash
# run_experiments.sh
# Usage: bash run_experiments.sh

YAML_DIR="rssm_configs/"
LOG_DIR="logs_rssm/"
mkdir -p $LOG_DIR

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
        wait $pid
    done
    echo "[BATCH DONE]"
}

run_yaml_modifier() {
    local latent=$1
    local k=$2

    python yaml_modifier.py \
        --yaml_dir $YAML_DIR \
        --latent $latent \
        --k $k
}

# ── Experiment grid ──────────────────────────────
LATENTS=(8 16 32 64)
K_VALUES=(50)
# SEED_VALUES=(0 1 2 3 4)
for k in "${K_VALUES[@]}"; do
    for latent in "${LATENTS[@]}"; do
        echo "==============================="
        echo "Running latent=$latent k=$k"
        echo "==============================="

        # Step 1 — modify all yamls
        run_yaml_modifier $latent $k

        # Step 2 — get all yamls
        yamls=($(ls $YAML_DIR/*.yaml))

        # Step 3 — run in batches of 5
        for ((i=0; i<${#yamls[@]}; i+=6)); do
            batch=("${yamls[@]:i:6}")
            echo "[BATCH $((i/6 + 1))] ${batch[@]}"
            run_batch "${batch[@]}"
        done

    done
done

echo "All experiments complete"