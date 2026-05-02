import sys
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Add root to sys.path
sys.path.append(os.getcwd())

from meta_bidding.train.ddrl.ddrl_trainer import LSTMTrainer

# ==========================================
# 1. 路径配置
# ==========================================
DA_MODEL_PATH = "logs/metabidding-ddrl/da_only/20260316-1658/950.pth"
RT_MODEL_PATH = "logs/metabidding-ddrl/rt_only/20260316-1743/50.pth"
OPTIMAL_RESULTS_PATH = "isolated_bidding_results.csv"
OUTPUT_CSV = "ddrl_vs_optimal_isolated.csv"
DEVICE = 'cuda:0'

# ==========================================
# 2. 评估核心函数 (采用你的 seq_len=1 手动推演范式)
# ==========================================
def evaluate_single_model(model_path, target_market):
    print(f"\nEvaluating mode: {target_market}")
    print(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    
    # 环境配置
    env_config = {
        'product': ['energy'], 
        'num_markets': 1, 
        'soc': 4.0, 
        'node': ['AECO'],
        'data_source': 'test', 
        'degradation_cost': 10.0,
        'device': DEVICE,
        'mode': target_market  # 注入模式，让底层网络知道当前处于哪种隔离模式
    }
    
    # Init Trainer (seq_len=1 按天评估)
    trainer = LSTMTrainer(batch_size=1, seq_len=1, device=DEVICE, env_config=env_config)
    trainer.load(model_path)
    trainer.eval()
    
    # 提取测试集日期
    test_indices = []
    dataset = trainer.env
    df = dataset.df_data
    if 'SETTLEMENTDATE' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['SETTLEMENTDATE']):
            df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
        for idx in dataset.sod_idx:
            date_val = df.iloc[idx]['SETTLEMENTDATE']
            if date_val.day >= 28:
                test_indices.append((idx, date_val.strftime('%Y-%m-%d')))
                
    print(f"Found {len(test_indices)} test days.")
    
    daily_profits = {}
    
    # 逐天推演
    for idx, date_str in tqdm(test_indices, desc=f"Testing {target_market}"):
        # 重置环境，强制设置当前指针和初始SOC
        trainer.env.reset()
        trainer.env._pcs = np.array([idx])
        trainer.env._soc = [torch.tensor([0.5], device=DEVICE)] # Initial SoC 0.5 * 4.0MWh = 2.0MWh
        
        with torch.no_grad():
            info = trainer.one_minibatch_step(verbose=True)
            
        # 提取数据 (参照可用的工作脚本)
        q_da_raw = info['da_action'][0].T # (24, 9)
        q_rt_raw = info['action'][:, 0, :] # (288, 9)
        p_rt_raw = info['lmp'][:, 0, :] # (288, 9)
        p_da_raw_short = info['lmp_da'][:, 0, :] # (24, 9)
        
        q_da_24 = q_da_raw[:, 0]
        q_rt_288 = q_rt_raw[:, 0]
        p_rt_288 = p_rt_raw[:, 0]
        p_da_24 = p_da_raw_short[:, 0]
        
        profit = 0.0
        deg_cost = env_config['degradation_cost'] # 10.0
        
        if target_market == 'da_only':
            for h in range(24):
                q = q_da_24[h].item()
                p = p_da_24[h].item()
                
                revenue = q * p  # 收入 (如果是充电，q 为负数，自动变为支出)
                degradation = deg_cost * q if q > 0 else 0.0 # 仅放电(q>0)时计算老化成本
                
                profit += (revenue - degradation)
                
        elif target_market == 'rt_only':
            dt = 1.0 / 12.0 # 5分钟 = 1/12 小时
            for t in range(288):
                q = q_rt_288[t].item()
                p = p_rt_288[t].item()
                
                revenue = q * p * dt
                degradation = deg_cost * q * dt if q > 0 else 0.0 # 仅放电时计算
                
                profit += (revenue - degradation)
                
        daily_profits[date_str] = profit
        
    return daily_profits

# ==========================================
# 3. 执行评估并与理论最优对比
# ==========================================
def main():
    if not os.path.exists(OPTIMAL_RESULTS_PATH):
        raise FileNotFoundError(f"Optimal results CSV not found: {OPTIMAL_RESULTS_PATH}")
        
    # 读取最优结果并剥离 Average 行
    df_optimal = pd.read_csv(OPTIMAL_RESULTS_PATH)
    df_optimal_days = df_optimal[df_optimal['Date'] != 'Average'].copy()
    
    # 分别运行两个模型的测试
    da_profits_dict = evaluate_single_model(DA_MODEL_PATH, 'da_only')
    rt_profits_dict = evaluate_single_model(RT_MODEL_PATH, 'rt_only')
    
    # 组装结果
    results = []
    for _, row in df_optimal_days.iterrows():
        date_str = row['Date']
        opt_da = row['Max_DA_Profit']
        opt_rt = row['Max_RT_Profit']
        
        # 提取模型收益，如果某天缺失则设为 NaN
        ddrl_da = da_profits_dict.get(date_str, np.nan)
        ddrl_rt = rt_profits_dict.get(date_str, np.nan)
        
        # 计算百分比
        da_pct = (ddrl_da / opt_da) * 100 if opt_da != 0 else 0
        rt_pct = (ddrl_rt / opt_rt) * 100 if opt_rt != 0 else 0
        
        results.append({
            'Date': date_str,
            'DA_Opt_Profit': opt_da,
            'DA_DDRL_Profit': ddrl_da,
            'DA_Percent (%)': da_pct,
            'RT_Opt_Profit': opt_rt,
            'RT_DDRL_Profit': ddrl_rt,
            'RT_Percent (%)': rt_pct
        })
        
    results_df = pd.DataFrame(results)
    
    # 追加 Average 行
    avg_row = results_df.mean(numeric_only=True)
    avg_row['Date'] = 'Average'
    results_df_final = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)
    
    # 打印结果
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.2f}'.format)
    print("\n" + "="*80)
    print("DDRL vs Optimal Bidding Evaluation Results")
    print("="*80)
    print(results_df_final)
    
    # 保存 CSV
    results_df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()