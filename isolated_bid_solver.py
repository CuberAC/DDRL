import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import argparse
import os

def solve_da_only_max(df_day, date_str, soc_max_mwh=4.0, power_max_mw=1.0, efficiency=0.9, initial_soc=0.5, degradation_cost=10.0):
    """
    仅参与日前市场 (DA-Only) 的最优化求解器
    变量按小时 (Hourly) 分布，目标函数求 sum(P_da * Q_da)
    """
    T = len(df_day)
    T_optim = min(T, 288)
    dt_rt = 1.0 / 12.0 # 价格信号虽为 5 min 精度，但 DA 动作在 1 小时内保持不变
    dt_da = 1.0
    eff_sqrt = np.sqrt(efficiency)
    
    # 获取 DA 价格
    if 'DA_RRP' in df_day.columns:
        da_prices_5min = df_day['DA_RRP'].values[:T_optim]
    else:
        da_cols = [c for c in df_day.columns if c.startswith('DA_')]
        da_prices_5min = df_day[da_cols[0]].values[:T_optim] if da_cols else df_day['RRP'].values[:T_optim]

    m = gp.Model(f"DA_Only_{date_str}")
    m.setParam('OutputFlag', 0)
    
    # DA 变量：24小时
    q_da_chg = m.addVars(24, lb=0, ub=power_max_mw, name="q_da_chg")
    q_da_dis = m.addVars(24, lb=0, ub=power_max_mw, name="q_da_dis")
    soc_da   = m.addVars(25, lb=0, ub=soc_max_mwh, name="soc_da")
    
    # 初始 SoC 约束
    init_energy = initial_soc * soc_max_mwh
    m.addConstr(soc_da[0] == init_energy, "InitSoC_DA")
    
    # SoC 按小时演变规律 (完全取决于 DA 计划)
    for h in range(24):
        m.addConstr(
            soc_da[h+1] == soc_da[h] + (q_da_chg[h] * eff_sqrt - q_da_dis[h] / eff_sqrt) * dt_da,
            f"SoC_Rule_DA_{h}"
        )
        
    # 目标函数：纯粹的 P_da * Q_da
    obj_expr = 0
    for t in range(T_optim):
        h = int(t / 12)
        if h >= 24: h = 23
        
        # 净出力 (放电 - 充电)
        q_da_net_h = q_da_dis[h] - q_da_chg[h]
        
        # 1. 市场电费收益 ($)
        revenue = q_da_net_h * da_prices_5min[t] * dt_rt
        
        # 2. 电池老化成本 ($)：仅在放电时产生
        degradation = degradation_cost * q_da_dis[h] * dt_rt
        
        # 3. 累加净利润
        obj_expr += (revenue - degradation)
        
    m.setObjective(obj_expr, GRB.MAXIMIZE)
    m.optimize()
    
    if m.Status != GRB.OPTIMAL:
        return None
    return m.ObjVal

def solve_rt_only_max(df_day, date_str, soc_max_mwh=4.0, power_max_mw=1.0, efficiency=0.9, initial_soc=0.5, degradation_cost=10.0):
    """
    仅参与实时市场 (RT-Only) 的最优化求解器
    变量按5分钟 (5-min) 分布，目标函数求 sum(P_rt * Q_rt)
    """
    T = len(df_day)
    T_optim = min(T, 288)
    dt_rt = 1.0 / 12.0
    eff_sqrt = np.sqrt(efficiency)
    
    # 获取 RT 价格
    rt_prices = df_day['RRP'].values[:T_optim]

    m = gp.Model(f"RT_Only_{date_str}")
    m.setParam('OutputFlag', 0)
    
    # RT 变量：288 个步长
    q_rt_chg = m.addVars(T_optim, lb=0, ub=power_max_mw, name="q_rt_chg")
    q_rt_dis = m.addVars(T_optim, lb=0, ub=power_max_mw, name="q_rt_dis")
    soc_rt   = m.addVars(T_optim+1, lb=0, ub=soc_max_mwh, name="soc_rt")
    
    # 初始 SoC 约束
    init_energy = initial_soc * soc_max_mwh
    m.addConstr(soc_rt[0] == init_energy, "InitSoC_RT")
    
    # SoC 按 5 分钟演变规律 (完全取决于 RT 执行)
    for t in range(T_optim):
        m.addConstr(
            soc_rt[t+1] == soc_rt[t] + (q_rt_chg[t] * eff_sqrt - q_rt_dis[t] / eff_sqrt) * dt_rt,
            f"SoC_Rule_RT_{t}"
        )
        
    # 目标函数：纯粹的 P_rt * Q_rt
    obj_expr = 0
    for t in range(T_optim):
        # 净出力
        q_rt_net_t = q_rt_dis[t] - q_rt_chg[t]

        # 1. 市场电费收益 ($)
        revenue = q_rt_net_t * rt_prices[t] * dt_rt

        # 2. 电池老化成本 ($)：仅在放电时产生
        degradation = degradation_cost * q_rt_dis[t] * dt_rt

        # 3. 累加净利润
        obj_expr += (revenue - degradation)
        
    m.setObjective(obj_expr, GRB.MAXIMIZE)
    m.optimize()
    
    if m.Status != GRB.OPTIMAL:
        return None
    return m.ObjVal


def main():
    # 替换为你实际的测试集数据路径
    data_path = "meta_bidding/data/pjm_data/pjm_price_test_dual_split.pkl"
    if not os.path.exists(data_path):
        print(f"Data not found at: {data_path}")
        return

    df = pd.read_pickle(data_path)
    
    # 时间处理
    if 'SETTLEMENTDATE' in df.columns:
        df['dt'] = pd.to_datetime(df['SETTLEMENTDATE'])
    else:
        df['dt'] = pd.to_datetime(df.index)
    
    df['date_str'] = df['dt'].dt.strftime('%Y-%m-%d')
    df['day_of_month'] = df['dt'].dt.day
    
    # 选取测试天数 (与原始代码一致)
    test_dates = sorted(df[df['day_of_month'] >= 28]['date_str'].unique())
    
    print(f"{'Date':<15} | {'Max DA-Only Profit ($)':<22} | {'Max RT-Only Profit ($)':<22}")
    print("-" * 65)
    
    results = []
    for date_str in test_dates:
        df_day = df[df['date_str'] == date_str].copy()
        
        # 1. 计算日前上限
        max_da = solve_da_only_max(df_day, date_str)
        # 2. 计算实时上限
        max_rt = solve_rt_only_max(df_day, date_str)
        
        if max_da is not None and max_rt is not None:
            results.append({'Date': date_str, 'Max_DA_Profit': max_da, 'Max_RT_Profit': max_rt})
            print(f"{date_str:<15} | {max_da:<22.2f} | {max_rt:<22.2f}")
        else:
            print(f"{date_str:<15} | {'Solver Failed':<22} | {'Solver Failed':<22}")

    # 将结果保存为 CSV
    if results:
        df_res = pd.DataFrame(results)
        
        # 计算平均值
        avg_da = df_res['Max_DA_Profit'].mean()
        avg_rt = df_res['Max_RT_Profit'].mean()
        
        print("-" * 65)
        print(f"{'Average':<15} | {avg_da:<22.2f} | {avg_rt:<22.2f}")
        
        # 增加一行 Average 用于保存到 CSV
        avg_row = pd.DataFrame([{'Date': 'Average', 'Max_DA_Profit': avg_da, 'Max_RT_Profit': avg_rt}])
        df_final = pd.concat([df_res, avg_row], ignore_index=True)
        
        # 导出为 CSV
        save_path = "isolated_bidding_results.csv"
        df_final.to_csv(save_path, index=False)
        print(f"\nOptimization Complete.")
        print(f"Results successfully saved to: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    main()