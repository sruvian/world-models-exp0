#!/bin/bash
# run_experiments.sh
# Usage: bash run_experiments.sh

YAML_DIR="trainer_configs/"
LOG_DIR="logs/cartpole"
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
LATENTS=(3 8 16 32)
K_VALUES=(1 5 15 50)

for latent in "${LATENTS[@]}"; do
    for k in "${K_VALUES[@]}"; do
        echo "==============================="
        echo "Running latent=$latent k=$k"
        echo "==============================="

        # Step 1 — modify all yamls
        run_yaml_modifier $latent $k

        # Step 2 — get all yamls
        yamls=($(ls $YAML_DIR/cartpole_*.yaml))

        # Step 3 — run in batches of 5
        for ((i=0; i<${#yamls[@]}; i+=5)); do
            batch=("${yamls[@]:i:5}")
            echo "[BATCH $((i/5 + 1))] ${batch[@]}"
            run_batch "${batch[@]}"
        done

    done
done

echo "All experiments complete"