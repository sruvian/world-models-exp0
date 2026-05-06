#!/bin/bash

CONFIGS=(
  "pendulum_g5.0_l2.0"
  "pendulum_g5.0_l10.0"
  "pendulum_g5.0_l18.0"
  "pendulum_g9.80_l2.0"
  "pendulum_g9.80_l10.0"
  "pendulum_g9.80_l18.0"
  "pendulum_g15.0_l2.0"
  "pendulum_g15.0_l10.0"
  "pendulum_g15.0_l18.0"
  "pendulum_combined"
)

mkdir -p logs

for CONFIG in "${CONFIGS[@]}"; do
  echo "Launching $CONFIG"
  nohup python main.py --yaml configs/${CONFIG}.yaml > logs/${CONFIG}.log 2>&1 &
done

echo "All 10 jobs launched"