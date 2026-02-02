import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import argparse
import tqdm

def solve_one_day(
    df_day,
    dt_steps_per_hour=12,
    soc_max_mwh=4.0, 
    power_max_mw=1.0, 
    eff_single=0.948, 
    degradation_cost=50.0, 
    initial_soc=0.5,
    resample_factor=3 
):
    """
    求解单日优化问题的辅助函数 (假设 Perfect Foresight)
    """
    # 原始数据提取
    rt_prices_raw = df_day['RRP'].values
    da_prices_raw = df_day['DA_RRP'].values
    
    # 确保数据长度足够重采样，不足则截断 (必须是 resample_factor 的倍数，如 3)
    trim_len = (len(rt_prices_raw) // resample_factor) * resample_factor
    if trim_len == 0: return None
    
    # 重采样: 5min -> 15min
    rt_prices = rt_prices_raw[:trim_len].reshape(-1, resample_factor).mean(axis=1)
    da_prices = da_prices_raw[:trim_len].reshape(-1, resample_factor).mean(axis=1)
    
    # 更新时间参数
    dt = (5.0 * resample_factor) / 60.0 # 15分钟 = 0.25 小时
    T = len(rt_prices)
    
    # 创建 Gurobi 模型
    m = gp.Model("OptimalBidding")
    m.setParam('OutputFlag', 0) 

    # 定义变量
    # 日前承诺与实时调度
    q_da = m.addVars(T, lb=-power_max_mw, ub=power_max_mw, name="q_da")
    q_rt = m.addVars(T, lb=-power_max_mw, ub=power_max_mw, name="q_rt")
    # SoC 状态 (实时 & 日前虚拟)
    soc_rt = m.addVars(T+1, lb=0, ub=soc_max_mwh, name="soc_rt")
    soc_da = m.addVars(T+1, lb=0, ub=soc_max_mwh, name="soc_da")
    
    # 辅助变量
    q_rt_dis = m.addVars(T, lb=0, ub=power_max_mw, name="q_rt_dis")
    q_rt_chg = m.addVars(T, lb=0, ub=power_max_mw, name="q_rt_chg")
    q_da_dis = m.addVars(T, lb=0, ub=power_max_mw, name="q_da_dis")
    q_da_chg = m.addVars(T, lb=0, ub=power_max_mw, name="q_da_chg")
    
    # 约束: 功率平衡
    m.addConstrs((q_rt[t] == q_rt_dis[t] - q_rt_chg[t] for t in range(T)), "link_q_rt")
    m.addConstrs((q_da[t] == q_da_dis[t] - q_da_chg[t] for t in range(T)), "link_q_da")
    
    # 约束: 初始 SoC
    m.addConstr(soc_rt[0] == initial_soc * soc_max_mwh, "init_soc_rt")
    m.addConstr(soc_da[0] == initial_soc * soc_max_mwh, "init_soc_da")
    
    # 约束: SoC 动态方程
    m.addConstrs((soc_rt[t+1] == soc_rt[t] - (q_rt_dis[t] / eff_single) * dt + (q_rt_chg[t] * eff_single) * dt for t in range(T)), "soc_update_rt")
    # 日前约束也是物理可行的，防止过度承诺
    m.addConstrs((soc_da[t+1] == soc_da[t] - (q_da_dis[t] / eff_single) * dt + (q_da_chg[t] * eff_single) * dt for t in range(T)), "soc_update_da")
    
    # 构建目标函数
    total_profit = 0
    
    for t in range(T):
        p_rt = rt_prices[t]
        p_da = da_prices[t]
        # 收益 = Q_da * P_da + (Q_rt - Q_da) * P_rt
        revenue_step = q_da[t] * p_da + (q_rt[t] - q_da[t]) * p_rt
        
        # 成本 = 老化单价 * 实时放电量
        cost_deg_step = degradation_cost * q_rt_dis[t]
        
        total_profit += (revenue_step - cost_deg_step) * dt
        
    m.setObjective(total_profit, GRB.MAXIMIZE)
    m.optimize()
    
    if m.status == GRB.OPTIMAL:
        # 提取结果计算指标 (还原到 $)
        q_da_val = np.array([q_da[t].x for t in range(T)])
        q_rt_val = np.array([q_rt[t].x for t in range(T)])
        q_rt_dis_val = np.array([q_rt_dis[t].x for t in range(T)])
        
        income_da = np.sum(q_da_val * da_prices) * dt
        income_rt = np.sum((q_rt_val - q_da_val) * rt_prices) * dt
        cost_degradation = np.sum(q_rt_dis_val * degradation_cost) * dt
        net_profit = income_da + income_rt - cost_degradation
        
        return {
            "net_profit": net_profit,
            "income_da": income_da,
            "income_rt": income_rt,
            "cost_degradation": cost_degradation
        }
    else:
        return None

def solve_optimal_bidding_year(
    price_data_path, 
    node='AECO', 
    soc_max_mwh=4.0, 
    power_max_mw=1.0, 
    efficiency=0.9, 
    degradation_cost=50.0, 
    initial_soc=0.5,
    output_csv="optimal_bidding_year_results.csv"
):
    """
    求解全年的最优策略并保存表格
    """
    print(f"正在加载数据: {price_data_path}...")
    try:
        df = pd.read_pickle(price_data_path)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    if 'REGIONID' in df.columns and node:
        df = df[df['REGIONID'] == node]
        print(f"已筛选节点 {node} 的数据，共 {len(df)} 行。")
    
    if len(df) == 0:
        print("未找到数据。")
        return

    # 检查日期列
    if 'SETTLEMENTDATE' not in df.columns:
        print("Error: SETTLEMENTDATE column not found.")
        return
        
    # 按天分组
    df['Date'] = df['SETTLEMENTDATE'].dt.date
    unique_dates = df['Date'].unique()
    unique_dates.sort() # 确保按时间顺序
    
    print(f"共发现 {len(unique_dates)} 天的数据。开始逐日计算...")
    print(f"参数设置: 老化成本 ${degradation_cost}/MWh, 效率 {efficiency}")
    
    eff_single = np.sqrt(efficiency)
    results = []
    
    # 进度条循环计算每一天
    for date in tqdm.tqdm(unique_dates):
        # 提取当天数据
        df_day = df[df['Date'] == date].reset_index(drop=True)
        
        # 简单检查数据完整性 (全天应该有 288 个点，允许少量缺失)
        if len(df_day) < 280: 
             continue

        # 求解当天
        res = solve_one_day(
            df_day,
            soc_max_mwh=soc_max_mwh,
            power_max_mw=power_max_mw,
            eff_single=eff_single,
            degradation_cost=degradation_cost,
            initial_soc=initial_soc,
            resample_factor=3 # 固定为15min间隔以加速
        )
        
        if res:
            res['date'] = date
            results.append(res)
            
    # 保存结果
    res_df = pd.DataFrame(results)
    # 调整列顺序
    cols = ['date', 'net_profit', 'income_da', 'income_rt', 'cost_degradation']
    res_df = res_df[cols]
    
    print(f"\n计算完成！前5行结果:")
    print(res_df.head())
    
    res_df.to_csv(output_csv, index=False)
    print(f"\n完整结果已保存至 {output_csv}")
    
    print("\n全年统计摘要 (Statistics):")
    print(res_df.describe())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PJM 全年电池储能最优投标计算器")
    parser.add_argument("--data_path", type=str, default="meta_bidding/data/pjm_data/pjm_price_train_dual.pkl", help="价格数据路径")
    parser.add_argument("--node", type=str, default="AECO", help="PJM 节点名称")
    parser.add_argument("--degradation_cost", type=float, default=50.0, help="电池老化成本 ($/MWh)")
    parser.add_argument("--output", type=str, default="optimal_bidding_year_results.csv", help="输出CSV文件名")
    
    args = parser.parse_args()
    
    solve_optimal_bidding_year(
        price_data_path=args.data_path,
        node=args.node,
        degradation_cost=args.degradation_cost,
        output_csv=args.output
    )
