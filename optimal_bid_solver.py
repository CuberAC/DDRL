import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import argparse
import matplotlib.pyplot as plt

# 设置 Matplotlib 后端，避免在无显示器环境下报错
plt.switch_backend('Agg')

def solve_optimal_bidding(
    price_data_path, 
    node='AECO', 
    soc_max_mwh=1.0, 
    power_max_mw=1.0, 
    efficiency=0.9, 
    degradation_cost=0, 
    initial_soc=0.5,
    horizon_hours=24
):
    """
    使用 Gurobi 求解基于完美价格预测的最优日前和实时投标策略。
    包括收益分解计算和绘图功能。
    """
    print(f"正在加载数据: {price_data_path}...")
    try:
        df = pd.read_pickle(price_data_path)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 按节点筛选数据
    if 'REGIONID' in df.columns and node:
        df = df[df['REGIONID'] == node]
        print(f"已筛选节点 {node} 的数据，共 {len(df)} 行。")
    
    if len(df) == 0:
        print("未找到数据。")
        return

    # 时间参数
    dt = 5.0 / 60.0 # 5分钟对应的小时数
    dt_steps_per_hour = 12
    T = int(horizon_hours * dt_steps_per_hour) # 总时间步数
    
    # 截取前 T 个时间步的数据 (原始 5min)
    # 为了适应 Gurobi 限制版 License (2000 变量限制), 我们将数据重采样到 15 分钟间隔
    # T_raw = horizon 24h * 12 = 288
    # T_new = horizon 24h * 4 = 96
    
    # 原始切片
    df_raw = df.iloc[:int(horizon_hours * 12)].copy().reset_index(drop=True)
    
    # 重采样处理
    # 假设数据索引是连续的 5min, 我们可以每 3 行取平均
    resample_factor = 3 # 5min -> 15min
    
    # 提取并重采样价格
    # 注意: RRP 和 DA_RRP 是时段价格, 取平均是合理的
    rt_prices_raw = df_raw['RRP'].values
    da_prices_raw = df_raw['DA_RRP'].values
    
    # Reshape and mean
    # 确保长度能被 3 整除
    trim_len = (len(rt_prices_raw) // resample_factor) * resample_factor
    rt_prices = rt_prices_raw[:trim_len].reshape(-1, resample_factor).mean(axis=1)
    da_prices = da_prices_raw[:trim_len].reshape(-1, resample_factor).mean(axis=1)
    
    # 更新时间参数
    dt = (5.0 * resample_factor) / 60.0 # 15分钟 = 0.25 小时
    T = len(rt_prices)
    
    print(f"数据已重采样为 {dt*60:.0f} 分钟间隔, 时间步数: {T}")
    
    # 效率参数转换为单程效率: sqrt(0.9) approx 0.948
    eff_single = np.sqrt(efficiency) 
    
    # 创建 Gurobi 模型
    m = gp.Model("OptimalBidding")
    m.setParam('OutputFlag', 0) # 关闭冗余输出，仅显示最终结果

    # 定义变量
    # q_da: 日前承诺电量 (MW)
    # q_rt: 实时实际调度电量 (MW)
    q_da = m.addVars(T, lb=-power_max_mw, ub=power_max_mw, name="q_da")
    q_rt = m.addVars(T, lb=-power_max_mw, ub=power_max_mw, name="q_rt")
    
    # soc: 电池荷电状态 (MWh) (实时)
    soc_rt = m.addVars(T+1, lb=0, ub=soc_max_mwh, name="soc_rt")
    # soc: 电池荷电状态 (MWh) (日前 - 用于确保计划可行性)
    soc_da = m.addVars(T+1, lb=0, ub=soc_max_mwh, name="soc_da")
    
    # 辅助变量：用于计算充电和放电的物理量
    # 实时
    q_rt_dis = m.addVars(T, lb=0, ub=power_max_mw, name="q_rt_dis")
    q_rt_chg = m.addVars(T, lb=0, ub=power_max_mw, name="q_rt_chg")
    # 日前 (分解变量以计算日前SoC)
    q_da_dis = m.addVars(T, lb=0, ub=power_max_mw, name="q_da_dis")
    q_da_chg = m.addVars(T, lb=0, ub=power_max_mw, name="q_da_chg")
    
    # 约束：实时功率平衡 Q_rt = Dis - Chg
    m.addConstrs((q_rt[t] == q_rt_dis[t] - q_rt_chg[t] for t in range(T)), "link_q_rt")
    # 约束：日前功率平衡 Q_da = Dis - Chg
    m.addConstrs((q_da[t] == q_da_dis[t] - q_da_chg[t] for t in range(T)), "link_q_da")
    
    # 约束：初始 SoC
    m.addConstr(soc_rt[0] == initial_soc * soc_max_mwh, "init_soc_rt")
    m.addConstr(soc_da[0] == initial_soc * soc_max_mwh, "init_soc_da")
    
    # 约束：SoC 动态方程 (实时)
    # SoC(t+1) = SoC(t) - 放电/效率*dt + 充电*效率*dt
    m.addConstrs((soc_rt[t+1] == soc_rt[t] - (q_rt_dis[t] / eff_single) * dt + (q_rt_chg[t] * eff_single) * dt for t in range(T)), "soc_update_rt")
    
    # 约束：SoC 动态方程 (日前 - 虚拟轨迹，确保承诺可行)
    m.addConstrs((soc_da[t+1] == soc_da[t] - (q_da_dis[t] / eff_single) * dt + (q_da_chg[t] * eff_single) * dt for t in range(T)), "soc_update_da")
    
    # 构建目标函数
    total_profit = 0
    
    for t in range(T):
        p_rt = rt_prices[t]
        p_da = da_prices[t]
        
        # 收益公式:
        # 日前结算 = Q_da * P_da
        # 实时结算 = (Q_rt - Q_da) * P_rt
        # 实际上 = Q_da * (P_da - P_rt) + Q_rt * P_rt
        revenue_step = q_da[t] * p_da + (q_rt[t] - q_da[t]) * p_rt
        
        # 电池老化成本 (仅针对实际发生的实时放电部分)
        cost_deg_step = degradation_cost * q_rt_dis[t]
        
        # 累加
        total_profit += (revenue_step - cost_deg_step) * dt
        
    m.setObjective(total_profit, GRB.MAXIMIZE)
    
    # 开始求解
    print("正在求解优化问题 (已添加日前物理约束)...")
    m.optimize()
    
    if m.status == GRB.OPTIMAL:
        print("\n=== 最优解已找到 ===")
        
        # 提取结果
        q_da_val = np.array([q_da[t].x for t in range(T)])
        q_rt_val = np.array([q_rt[t].x for t in range(T)])
        q_rt_dis_val = np.array([q_rt_dis[t].x for t in range(T)])
        soc_val = np.array([soc_rt[t].x for t in range(T+1)])
        soc_da_val = np.array([soc_da[t].x for t in range(T+1)])
        
        # 计算详细收益指标
        income_da = np.sum(q_da_val * da_prices) * dt
        income_rt = np.sum((q_rt_val - q_da_val) * rt_prices) * dt
        cost_degradation = np.sum(q_rt_dis_val * degradation_cost) * dt
        net_profit = income_da + income_rt - cost_degradation
        
        # 打印控制台报告
        print(f"{'指标':<20} | {'数值':>10}")
        print("-" * 35)
        print(f"{'总净利润 (Net Profit)':<20} | ${net_profit:10.2f}")
        print(f"{'日前市场收益 (DA)':<20} | ${income_da:10.2f}")
        print(f"{'实时市场收益 (RT)':<20} | ${income_rt:10.2f}")
        print(f"{'电池老化成本 (Cost)':<20} | ${cost_degradation:10.2f}")
        print("-" * 35)
        
        # 绘图 (使用英文标签以兼容无中文字库环境)
        plt.figure(figsize=(12, 12))
        
        # 图1: 电价对比
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(rt_prices, label='RT Price', color='#ff7f0e', alpha=0.8)
        ax1.plot(da_prices, label='DA Price', color='#1f77b4', linestyle='--', alpha=0.8)
        ax1.set_ylabel("Price ($/MWh)")
        ax1.set_title(f"Price Profile (Node: {node})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2: 投标与调度
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(q_rt_val, label='RT Dispatch', color='#2ca02c', linewidth=2)
        ax2.plot(q_da_val, label='DA Plan', color='#d62728', linestyle='--', linewidth=2)
        ax2.set_ylabel("Power (MW)")
        ax2.set_title("Day-Ahead Plan vs Real-Time Dispatch")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 图3: 荷电状态 SoC
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(soc_val, label='RT SoC', color='#9467bd', linewidth=2)
        ax3.plot(soc_da_val, label='DA Planned SoC', color='#d62728', linestyle='--', linewidth=1.5, alpha=0.7)
        ax3.axhline(soc_max_mwh, color='k', linestyle=':', alpha=0.5, label='Max SoC')
        ax3.axhline(0, color='k', linestyle=':', alpha=0.5, label='Min SoC')
        ax3.set_ylabel("Energy (MWh)")
        ax3.set_xlabel("Time Step (5-min intervals)")
        ax3.set_title("State of Charge (SoC)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = "optimal_bidding_plot.png"
        plt.savefig(save_path, dpi=100)
        print(f"\n图表已保存至: {save_path}")
        
    else:
        print("优化求解失败。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PJM 电池储能最优投标计算器 (标准答案)")
    parser.add_argument("--data_path", type=str, default="meta_bidding/data/pjm_data/pjm_price_train_dual.pkl", help="价格数据路径")
    parser.add_argument("--node", type=str, default="AECO", help="PJM 节点名称")
    parser.add_argument("--days", type=int, default=1, help="计算天数")
    parser.add_argument("--degradation_cost", type=float, default=50.0, help="电池老化成本 ($/MWh)")
    
    args = parser.parse_args()
    
    solve_optimal_bidding(
        price_data_path=args.data_path,
        node=args.node,
        horizon_hours=24 * args.days,
        degradation_cost=args.degradation_cost
    )
