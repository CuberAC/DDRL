#!/bin/bash
set -e

# Parameters
EPOCHS=2000
DEGRADATION_COST=10
PRODUCT="energy"
NODE="AECO"
SOC=4
LOG_BASE_DIR="/root/DDRL/logs/metabidding-ddrl/meta-ddrl-default"

echo "Starting training for $EPOCHS epochs..."
python meta_bidding/train/ddrl/train.py \
    --product $PRODUCT \
    --soc $SOC \
    --node $NODE \
    --total_epoches $EPOCHS \
    --degradation_cost $DEGRADATION_COST

# Find the latest directory created in logs
# Use ls -td to sort by time, head -1 to get the newest
LATEST_DIR=$(ls -td $LOG_BASE_DIR/*/ | head -1)

echo "Training finished. Latest directory is: $LATEST_DIR"

echo "Starting evaluation..."
python evaluate_checkpoints.py \
    --model_dir $LATEST_DIR \
    --degradation_cost $DEGRADATION_COST \
    --product $PRODUCT \
    --node $NODE \
    --soc $SOC

echo "All done!"
