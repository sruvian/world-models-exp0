#!/bin/bash

set -u

LOG_DIR="logs_rssm"
YAML_DIR="rssm_generated"
CKPT_DIR="model_saves_rssm"

mkdir -p "$LOG_DIR"

MAX_JOBS=$(( $(nproc) - 2 ))

PYTHON="$HOME/world-models-exp0/WSim/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Could not find Python:"
    echo "$PYTHON"
    exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE

mapfile -t yamls < <(find "$YAML_DIR" -name "*.yaml" | sort -r)

echo "===================================="
echo "Found ${#yamls[@]} YAMLs"
echo "Running with $MAX_JOBS workers"
echo "===================================="

for yaml in "${yamls[@]}"; do

    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        wait -n
    done

    name=$(basename "$yaml" .yaml)

    ##########################################################
    # Environment
    ##########################################################

    if [[ $name == pendulum_* ]]; then
        ENV="pendulum"
        base=${name#pendulum_}
    else
        ENV="cartpole"
        base=${name#cartpole_}
    fi

    ##########################################################
    # Impulse or normal?
    ##########################################################

    if [[ $base == *_imp_* ]]; then
        SEARCH_DIR="$CKPT_DIR/impulse_policy/$ENV"
    else
        SEARCH_DIR="$CKPT_DIR/$ENV"
    fi

    ##########################################################
    # Parse yaml name
    ##########################################################

    probe=$(echo "$base" | sed -E 's/_k0*[0-9]+_lat[0-9]+$//')

    # checkpoint filenames DO NOT contain "_imp"
    probe=${probe%_imp}

    rollout=$(echo "$base" | sed -E 's/.*_k0*([0-9]+)_lat.*/\1/')
    latent=$(echo "$base" | sed -E 's/.*_lat([0-9]+).*/\1/')

    checkpoint="${SEARCH_DIR}/model_WorldModelRSSM_0_${probe}_k${rollout}_Linear_steps100000_latent${latent}_beta0.1.pt"

    ##########################################################
    # Skip if checkpoint exists
    ##########################################################

    if [[ -f "$checkpoint" ]]; then
        echo "[SKIP ] $name"
        continue
    fi

    ##########################################################
    # Launch
    ##########################################################

    (
        echo "[START] $name"

        "$PYTHON" main.py \
            --yaml "$yaml" \
            > "$LOG_DIR/${name}.log" 2>&1

        if [ $? -eq 0 ]; then
            echo "[DONE ] $name"
        else
            echo "[FAIL ] $name"
        fi

    ) &

done

wait

echo
echo "===================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "===================================="