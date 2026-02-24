import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import argparse
import matplotlib.pyplot as plt
import os
import sys

# Set Matplotlib backend
plt.switch_backend('Agg')

def solve_optimal_bidding_day(
    df_day, 
    date_str,
    save_dir,
    node='AECO', 
    soc_max_mwh=4.0, 
    power_max_mw=1.0, 
    efficiency=0.9, 
    initial_soc=0.5
):
    """
    Solve optimal bidding for a single day (24 hours, 288 steps).
    Includes strictly feasible Day-Ahead schedule constraints.
    """
    T = len(df_day)
    # Usually T should be 288 steps (5 min)
    if T < 280:
        print(f"[{date_str}] Warning: Data length is {T}, expected ~288. Skipping.")
        return None
        
    T_optim = 288 # Standardize to 288 steps
    if T > T_optim: T_optim = 288 # Clip if larger
    if T < T_optim: T_optim = T # Clip if smaller (though we warned)
    
    # Constants
    dt_rt = 1.0 / 12.0 # RT step: 5 minutes in hours
    dt_da = 1.0        # DA step: 1 hour in hours
    eff_sqrt = np.sqrt(efficiency) # One-way efficiency
    
    # Prices (Align to T_optim)
    rt_prices = df_day['RRP'].values[:T_optim]
    
    if 'DA_RRP' in df_day.columns:
        da_prices_5min = df_day['DA_RRP'].values[:T_optim]
    else:
        # Fallback
        da_cols = [c for c in df_day.columns if c.startswith('DA_')]
        if da_cols:
            da_prices_5min = df_day[da_cols[0]].values[:T_optim]
        else:
            da_prices_5min = rt_prices

    # --- Gurobi Model ---
    m = gp.Model(f"OptimalBidding_{date_str}")
    m.setParam('OutputFlag', 0)
    
    # --- Variables ---
    
    # 1. Day-Ahead (Hourly)
    # To correctly track SoC, we need split Charge/Discharge variables for DA
    q_da_chg = m.addVars(24, lb=0, ub=power_max_mw, name="q_da_chg")
    q_da_dis = m.addVars(24, lb=0, ub=power_max_mw, name="q_da_dis")
    soc_da   = m.addVars(25, lb=0, ub=soc_max_mwh, name="soc_da") # 0 to 24 indices
    
    # 2. Real-Time (5-min)
    q_rt_chg = m.addVars(T_optim, lb=0, ub=power_max_mw, name="q_rt_chg")
    q_rt_dis = m.addVars(T_optim, lb=0, ub=power_max_mw, name="q_rt_dis")
    soc_rt   = m.addVars(T_optim+1, lb=0, ub=soc_max_mwh, name="soc_rt")
    
    # --- Constraints ---
    
    # A. Initial State
    init_energy = initial_soc * soc_max_mwh
    m.addConstr(soc_da[0] == init_energy, "InitSoC_DA")
    m.addConstr(soc_rt[0] == init_energy, "InitSoC_RT")
    
    # B. Day-Ahead Constraints (Hourly)
    for h in range(24):
        # SoC Evolution: S_{h+1} = S_h + (Chg*Eff - Dis/Eff) * dt_da
        # UNIT CHECK: MW * h * 1 = MWh. Correct.
        m.addConstr(
            soc_da[h+1] == soc_da[h] + (q_da_chg[h] * eff_sqrt - q_da_dis[h] / eff_sqrt) * dt_da,
            f"SoC_Rule_DA_{h}"
        )
    
    # C. Real-Time Constraints (5-min)
    # RT must track actual SoC evolution with 5-min steps
    for t in range(T_optim):
        # SoC Evolution
        m.addConstr(
            soc_rt[t+1] == soc_rt[t] + (q_rt_chg[t] * eff_sqrt - q_rt_dis[t] / eff_sqrt) * dt_rt,
            f"SoC_Rule_RT_{t}"
        )
        
    # --- Objective Function ---
    obj_expr = 0
    
    for t in range(T_optim):
        h = int(t / 12)
        if h >= 24: h = 23
        
        # DA Quantity for present hourly block
        q_da_net_h = q_da_dis[h] - q_da_chg[h]
        
        # RT Quantity for present 5-min interval
        q_rt_net_t = q_rt_dis[t] - q_rt_chg[t]
        
        # 1. DA Revenue
        # DA settles hourly. We accrue 1/12th per step to match loop structure.
        # This is mathematically equivalent to sum(Q_DA_h * P_DA_h * 1h) because sum(dt_rt over hour) = 1h.
        rev_da_step = q_da_net_h * da_prices_5min[t] * dt_rt
        
        # 2. RT Deviation Revenue
        # Deviation = RT_Net - DA_Net
        rev_rt_step = (q_rt_net_t - q_da_net_h) * rt_prices[t] * dt_rt
        
        obj_expr += rev_da_step + rev_rt_step
        
    m.setObjective(obj_expr, GRB.MAXIMIZE)
    
    # Optimize
    m.optimize()
    
    if m.Status != GRB.OPTIMAL:
        print(f"[{date_str}] Solver Failed. Status: {m.Status}")
        return None
        
    # --- Results extraction ---
    total_rev = m.ObjVal
    
    def get_vals(vars_dict, n):
        return np.array([vars_dict[i].x for i in range(n)])
    
    q_da_chg_h = get_vals(q_da_chg, 24)
    q_da_dis_h = get_vals(q_da_dis, 24)
    q_da_net_h = q_da_dis_h - q_da_chg_h
    soc_da_res_hourly = get_vals(soc_da, 25)
    
    # Interpolate DA SoC (Linear) for plotting
    soc_da_res_5min = []
    for h in range(24):
        start_s = soc_da_res_hourly[h]
        end_s = soc_da_res_hourly[h+1]
        lin = np.linspace(start_s, end_s, 13)[:-1]
        soc_da_res_5min.append(lin)
    soc_da_res_5min = np.concatenate(soc_da_res_5min)

    # Expand DA Power (Step)
    q_da_net_5min = np.repeat(q_da_net_h, 12)

    q_rt_chg_vals = get_vals(q_rt_chg, T_optim)
    q_rt_dis_vals = get_vals(q_rt_dis, T_optim)
    q_rt_net_vals = q_rt_dis_vals - q_rt_chg_vals
    soc_rt_res = get_vals(soc_rt, T_optim+1)
    
    da_rev_sum = 0
    rt_rev_sum = 0
    
    # Careful Summation for exact numbers
    for t in range(T_optim):
        # da_rev_sum += q_da_net_5min[t] * da_prices_5min[t] * dt_rt
        # rt_rev_sum += (q_rt_net_vals[t] - q_da_net_5min[t]) * rt_prices[t] * dt_rt
        
        # New Formula: DA = Arbitrage (Q_da * (P_da - P_rt)), RT = Physical (Q_rt * P_rt)
        da_rev_sum += q_da_net_5min[t] * (da_prices_5min[t] - rt_prices[t]) * dt_rt
        rt_rev_sum += q_rt_net_vals[t] * rt_prices[t] * dt_rt

    # Ensure length match for plotting (sometimes T_optim < 288)
    plot_len = min(len(soc_da_res_5min), len(soc_rt_res))
    time_axis = np.arange(plot_len) * dt_rt

    # --- Plotting ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # 1. Prices
    axes[0].step(time_axis, da_prices_5min[:plot_len], where='post', label='DA Price', color='blue', linestyle='--')
    axes[0].plot(time_axis, rt_prices[:plot_len], label='RT Price', color='orange', alpha=0.7)
    axes[0].set_ylabel("Price ($/MWh)")
    axes[0].set_title(f"Optimization Results: {date_str} (Profit: ${total_rev:.0f})")
    axes[0].legend(loc='upper right')
    
    # 2. Power
    axes[1].step(time_axis, q_da_net_5min[:plot_len], where='post', label='DA Bid (MW)', color='green', linewidth=2)
    axes[1].plot(time_axis, q_rt_net_vals[:plot_len], label='RT Dispatch (MW)', color='red', alpha=0.5)
    axes[1].set_ylabel("Power (MW)")
    axes[1].legend(loc='upper right')
    
    # 3. SoC (Comparison)
    axes[2].plot(time_axis, soc_da_res_5min[:plot_len], label='DA Virtual SoC', color='green', linestyle='--', linewidth=2)
    axes[2].plot(time_axis, soc_rt_res[:plot_len], label='RT Actual SoC', color='purple', alpha=0.8)
    axes[2].plot(time_axis, [soc_max_mwh]*plot_len, 'k--', alpha=0.3, label='Max SoC')
    axes[2].plot(time_axis, [0]*plot_len, 'k--', alpha=0.3)
    axes[2].set_ylabel("Energy (MWh)")
    axes[2].set_xlabel("Hours")
    axes[2].legend(loc='upper right')
    
    plot_path = os.path.join(save_dir, f"{date_str}_optimal.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return {
        'Date': date_str,
        'Day-Ahead Revenue ($)': da_rev_sum,
        'Real-Time Revenue ($)': rt_rev_sum, 
        'Total Revenue ($)': total_rev,
        'net_profit': total_rev
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=str, default='AECO', help="Node ID")
    parser.add_argument("--save_dir", type=str, default='eval_results_custom/optimal_bidding')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    data_path = "/root/DDRL/meta_bidding/data/pjm_data/pjm_price_test_dual_split.pkl"
    if not os.path.exists(data_path):
        print("Data not found.")
        return

    df = pd.read_pickle(data_path)
    if 'REGIONID' in df.columns:
        df = df[df['REGIONID'] == args.node]
    
    if 'SETTLEMENTDATE' in df.columns:
        df['dt'] = pd.to_datetime(df['SETTLEMENTDATE'])
    else:
        df['dt'] = pd.to_datetime(df.index)
    
    df['date_str'] = df['dt'].dt.strftime('%Y-%m-%d')
    df['day_of_month'] = df['dt'].dt.day
    
    test_dates = sorted(df[df['day_of_month'] >= 28]['date_str'].unique())
    print(f"Found {len(test_dates)} dates.")
    
    results = []
    for date_str in test_dates:
        print(f"Optimizing {date_str}...")
        df_day = df[df['date_str'] == date_str].copy()
        res = solve_optimal_bidding_day(
            df_day, date_str, args.save_dir, 
            node=args.node,
            soc_max_mwh=4.0, 
            power_max_mw=1.0
        )
        if res: results.append(res)
            
    if results:
        df_res = pd.DataFrame(results)
        # Average
        avg_row = df_res.mean(numeric_only=True)
        avg_row['Date'] = 'Average'
        df_res = pd.concat([df_res, pd.DataFrame([avg_row])], ignore_index=True)
        
        path1 = os.path.join(args.save_dir, "daily_performance_summary.csv")
        path2 = "/root/DDRL/optimal_bidding_year_results.csv"
        df_res.to_csv(path1, index=False)
        df_res.to_csv(path2, index=False)
        
        print("\nOptimization Complete.")
        print(f"Saved: {path1}")
        print(f"Saved: {path2}")
        print("Average Profit: {:.2f}".format(avg_row['Total Revenue ($)']))
    else:
        print("No results.")

if __name__ == "__main__":
    main()
