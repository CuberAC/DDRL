# Physics Informed Multi-Market Bidding of Energy Storage Based on Deep Differentiable Reinforcement Learning

To install the required packages, you can use the following command:

```bash
pip install -r requirements.txt
pip install -e .
```

To run the code, you can use the following command:

```bash
python meta_bidding/train/ddrl/train.py --product energy --soc 4 --node AECO --total_epoches 200
```

To test the trained agent, you can use the following command:

```bash
# Change the variable `ckpt_list` in the `scan_ddrl_aemo.py` file to checkpoint path.
python meta_bidding/train/ddrl/scan_ddrl_aemo.py
```