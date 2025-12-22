import sys
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

# 将当前目录添加到 sys.path 以便导入模块
sys.path.append(os.getcwd())

from meta_bidding.train.ddrl.ddrl_trainer import LSTMTrainer

def main():
    # 硬编码路径，指向用户指定的模型
    # 注意：这里假设用户指的是 logs/metabidding-ddrl/meta-ddrl-default/20251222-1053/150.pth
    # 如果有多个文件夹，请根据实际情况修改下面的 log_dir
    log_dir = "logs/metabidding-ddrl/meta-ddrl-default/20251222-1002"
    model_name = "150.pth"
    
    param_path = os.path.join(log_dir, "param.json")
    model_path = os.path.join(log_dir, model_name)
    
    # 检查文件是否存在
    if not os.path.exists(param_path):
        print(f"Error: Config file not found at {param_path}")
        base_log_dir = "logs/metabidding-ddrl/meta-ddrl-default/"
        candidates = [d for d in os.listdir(base_log_dir) if d.startswith("20251222") and os.path.exists(os.path.join(base_log_dir, d, model_name))]
        if candidates:
            print(f"Found alternative directories with {model_name}: {candidates}")
            log_dir = os.path.join(base_log_dir, candidates[-1]) # 使用最后一个（通常是最新的）
            param_path = os.path.join(log_dir, "param.json")
            model_path = os.path.join(log_dir, model_name)
            print(f"Using {log_dir} instead.")
        else:
            return

    print(f"Loading config from {param_path}")
    with open(param_path, 'r') as f:
        config = json.load(f)
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 修改配置以进行单日随机评估
    config['batch_size'] = 1      # 只运行一个环境实例
    config['eps_len'] = 1         # 只运行一天 (episode length = 1 day)
    config['device'] = device
    
    # 确保 env_config 包含所有必要的参数
    env_config = config.copy()
    
    print("Initializing Trainer and Environment...")
    # 初始化 Trainer
    trainer = LSTMTrainer(
        batch_size=config['batch_size'],
        seq_len=config['eps_len'],
        learning_rate=config['lr'],
        device=device,
        env_config=env_config
    )
    
    # 加载模型权重
    print(f"Loading model weights from {model_path}...")
    try:
        trainer.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    trainer.eval() # 切换到评估模式
    
    # 运行评估
    print("Running simulation for 1 random day...")
    with torch.no_grad():
        # evaluate_eps 内部会调用 reset()，这会随机选择一天的数据
        results = trainer.evaluate_eps()
        
    print(f"Simulation complete.")
    print(f"Mean Profit: {results['mean_profit']:.4f}")
    
    # 绘图
    output_dir = "eval_results_custom"
    plot_results(results, output_dir)

def plot_results(results, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 提取数据 (Batch 0, Day 0)
    # results['rt_ction'] shape: (Total_Steps, Batch, 9) -> (288, 1, 9)
    steps = 288
    batch_idx = 0
    
    # 确保数据长度足够
    if results['rt_action'].shape[0] < steps:
        print("Warning: Not enough steps for a full day plot.")
        steps = results['rt_action'].shape[0]

    rt_action = results['rt_action'][:steps, batch_idx, :]
    soc = results['soc'][:steps, batch_idx]
    lmp = results['lmp'][:steps, batch_idx, :]
    lmp_da = results['lmp_da'][:steps, batch_idx, :]
    
    # 处理 reward 维度
    if results['reward'].ndim == 3: # (Total_Steps, Batch, Markets)
        reward = np.sum(results['reward'][:steps, batch_idx, :], axis=1)
    else: # (Total_Steps, Batch)
        reward = results['reward'][:steps, batch_idx]
    
    # 创建图表
    fig, axes = plt.subplots(5, 1, figsize=(12, 20), sharex=True)
    
    # 1. Actions (Energy)
    axes[0].plot(rt_action[:, 0], label='RT Energy Action', color='blue')
    if 'da_action' in results and results['da_action'] is not None:
        # da_action shape: (Days, Batch, 9, 24) -> (1, 1, 9, 24)
        da_action = results['da_action'][0, batch_idx, :, :] # (9, 24)
        da_action_energy = np.repeat(da_action[0, :], 12) # 扩展到 288 个点
        axes[0].plot(da_action_energy, label='DA Energy Plan', color='red', linestyle='--', alpha=0.7)
    axes[0].set_ylabel('Power (MW)')
    axes[0].set_title('Energy Market Actions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. SoC
    axes[1].plot(soc, label='SoC', color='green')
    axes[1].set_ylabel('SoC (0-1)')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_title('State of Charge')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Prices
    axes[2].plot(lmp[:, 0], label='RT Price', color='orange')
    axes[2].plot(lmp_da[:, 0], label='DA Price', color='cyan', linestyle='--')
    axes[2].set_ylabel('Price ($/MWh)')
    axes[2].set_title('Energy Prices')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. Cumulative Reward
    cum_reward = np.cumsum(reward)
    axes[3].plot(cum_reward, label='Cumulative Reward', color='purple')
    axes[3].set_ylabel('Reward ($)')
    axes[3].set_title('Cumulative Reward')
    axes[3].grid(True, alpha=0.3)

    # 5. FCAS Actions (Reg Up/Down)
    axes[4].plot(rt_action[:, 1], label='Reg Up', alpha=0.7)
    axes[4].plot(rt_action[:, 2], label='Reg Down', alpha=0.7)
    axes[4].set_ylabel('Power (MW)')
    axes[4].set_title('FCAS Regulation Actions')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    axes[4].set_xlabel('Time Step (5-min)')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "daily_performance.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()
