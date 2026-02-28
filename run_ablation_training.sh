#!/bin/bash
echo "Starting DA Only Training..."
python meta_bidding/train/ddrl/train.py --product energy --soc 4 --node AECO --total_epoches 500 --mode da_only --exp_name da_only_500

echo "Starting RT Only Training..."
python meta_bidding/train/ddrl/train.py --product energy --soc 4 --node AECO --total_epoches 500 --mode rt_only --exp_name rt_only_500

echo "Ablation Training Complete."
