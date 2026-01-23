import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import os
import argparse
import matplotlib.pyplot as plt

def solve_optimal_bidding(
    price_data_path, 
    node='AECO', 
    soc_max_mwh=4.0, 
    power_max_mw=1.0, 
    efficiency=0.9, # Round trip efficiency sqrt(0.9)*sqrt(0.9) = 0.9? Code uses sqrt(0.9) for single trip
    degradation_cost=50.0, # $/MWh-cycle (approx)
    initial_soc=0.5,
    horizon_hours=24,
    product=['energy']
):
    """
    Solves the optimal bidding strategy with perfect foresight using Gurobi.
    """
    print(f"Loading data from {price_data_path}...")
    try:
        df = pd.read_pickle(price_data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Filter by node if present
    if 'REGIONID' in df.columns and node:
        df = df[df['REGIONID'] == node]
        print(f"Filtered data for node {node}. Rows: {len(df)}")
    
    if len(df) == 0:
        print("No data found.")
        return

    # Constants
    dt = 5.0 / 60.0 # 5 minutes in hours
    T = int(horizon_hours * 60 / 5) # Time steps
    
    # Slice first day/horizon for demonstration
    # In a real scenario, this would loop over days or be a rolling horizon
    # For "standard answer", we probably want to solve for a specific period used in evaluation.
    # Let's verify data structure first.
    # The file contains 5-min intervals.
    
    # We will pick the first T steps
    df_slice = df.iloc[:T].copy().reset_index(drop=True)
    
    # Extract prices
    # RT Prices
    rt_prices = {}
    da_prices = {}
    
    # Market map based on MetaDataset logic
    # Energy
    rt_prices['energy'] = df_slice['RRP'].values
    da_prices['energy'] = df_slice['DA_RRP'].values
    
    # Regulation
    # Note: Reg prices in PJM data seem to be capacity prices? Or movement?
    # MetaDataset uses: (0.25*regup_action_rt*lmps_rt[:,0] - 0.25*regdown_action_rt*lmps_rt[:,0]) implies Energy adjustment?
    # Wait, MetaDataset logic for Reg (AS) revenue:
    # settlement(as_rt[i], as_da[i], lmps_rt[:,i+1], lmps_da[:,i+1])
    # The AS prices are in columns like 'RAISEREGRRP', etc.
    
    # Let's map markets if requested
    use_reg = 'regulation' in product
    use_res = 'reserve' in product
    
    # Efficiency params
    eff_single = np.sqrt(efficiency) # 0.948
    
    # Create Gurobi Model
    m = gp.Model("OptimalBidding")
    m.setParam('OutputFlag', 1)

    # Variables
    # q_da: Day-Ahead Commitment (MW)
    # q_rt: Real-Time Dispatch (MW) matches the actual physical flow? 
    # In settlement formulation: Rev = Q_da * P_da + (Q_rt - Q_da) * P_rt
    # Where Q_rt is what actually happens physically.
    
    # Markets:
    # 0: Energy (Positive = Discharge, Negative = Charge)
    # 1: Reg Up
    # 2: Reg Down
    # ... Reserves
    
    # We simplify to Energy for now as base, add others if needed.
    # Energy Variables
    q_da = m.addVars(T, lb=-power_max_mw, ub=power_max_mw, name="q_da")
    q_rt = m.addVars(T, lb=-power_max_mw, ub=power_max_mw, name="q_rt")
    
    # SoC Variables (0 to 1 normalized, or MWh)
    # Let's use MWh
    soc = m.addVars(T+1, lb=0, ub=soc_max_mwh, name="soc")
    
    # Auxiliary variables for degradation cost (Absolute value of power)
    # Degradation = Cost * |Power| * dt? 
    # MetaDataset degradation: - self.DEGRATIO*self.MAXP*(energy_action_rt*(energy_action_rt>0) + 0.25*regup_action_rt)
    # It seems to penalize discharge and RegUp?
    # Standard battery degradation model usually proportional to throughput (cycle aging).
    # Let's strictly follow MetaDataset logic if possible to be comparable.
    # MetaDataset: reward_degradation = - DEGRATIO * MAXP * (E_rt * (E_rt>0) + 0.25 * RegUp_rt)
    # It only penalizes DISCHARGING energy? That's common if simplified.
    
    # Let's model split for discharge/charge to handle efficiency and cost
    q_rt_dis = m.addVars(T, lb=0, ub=power_max_mw, name="q_rt_dis")
    q_rt_chg = m.addVars(T, lb=0, ub=power_max_mw, name="q_rt_chg")
    
    m.addConstrs((q_rt[t] == q_rt_dis[t] - q_rt_chg[t] for t in range(T)), "link_q_rt")
    
    # Initial SoC
    m.addConstr(soc[0] == initial_soc * soc_max_mwh, "init_soc")
    
    # SoC Dynamics
    # SoC(t+1) = SoC(t) - Discharge * dt / eff + Charge * dt * eff (Wait, Discharge drains SoC)
    # Discharging (Output > 0): Drains Energy/Efficiency (losses occur inside?) or Energy * 1?
    # MetaDataset: 
    # discharge_soc_action = energy_action_rt*(energy_action_rt>0) + ...
    # charge_soc_action = energy_action_rt*(energy_action_rt<=0) ...
    # new_soc = soc - MAXPRTRATIO * (discharge_soc_action + charge_soc_action)
    # MAXPRTRATIO = MAXP / MAXSOC / 12 (12 steps per hour = dt) -> P_MW / E_MWh * (5/60)
    # Logic in code:
    # discharge_soc_action = energy (>0) 
    # charge_soc_action = energy (<0)
    # But wait, logic line 427: discharge_soc_action = ... + 0.25*regup/EFFICIENCY
    # charge_soc_action = ... - 0.25*regdown/EFFICIENCY
    # Actually wait.
    # Line 427: discharge_soc_action = energy_action_rt*(energy_action_rt>0) + ...
    # Line 428: charge_soc_action = energy_action_rt*(energy_action_rt<=0) - ... (minus negative = plus positive magnitude)
    # Wait, efficiency division?
    # Re-reading MetaDataset calc:
    # discharge_soc_action = energy > 0 ... / EFFICIENCY (line 427 in edited versions? Or original?)
    # Original snippet provided earlier: 
    # discharge_soc_action = energy_action_rt*(energy_action_rt>0) + 0.25*regup_action_rt/self.EFFICIENCY
    # charge_soc_action = energy_action_rt*(energy_action_rt<=0) - 0.25*regdown_action_rt/self.EFFICIENCY
    
    # It seems in MetaDataset.py provided:
    # discharge_soc_action = energy_action_rt*(energy_action_rt>0) + 0.25*regup_action_rt/self.EFFICIENCY
    # It divides by efficiency for AS?
    # What about Energy? Checks again.
    # Line 433 (approx): new_soc = self._soc[-1] - self.MAXPRTRATIO * total_soc_discharge_action
    
    # Let's use simpler standard battery model:
    # E(t+1) = E(t) - (q_dis / eff) * dt + (q_chg * eff) * dt
    # If MetaDataset logic is different, result might differ slightly.
    # Assuming standard:
    
    m.addConstrs((soc[t+1] == soc[t] - (q_rt_dis[t] / eff_single) * dt + (q_rt_chg[t] * eff_single) * dt for t in range(T)), "soc_update")
    
    # Objective Function
    obj = 0
    
    # 1. Energy Market Revenue
    # R_E = Q_da * P_da + (Q_rt - Q_da) * P_rt = Q_da * (P_da - P_rt) + Q_rt * P_rt
    # We maximize this sum over T
    for t in range(T):
        p_rt = rt_prices['energy'][t]
        p_da = da_prices['energy'][t]
        
        rev_energy = q_da[t] * (p_da - p_rt) + q_rt[t] * p_rt
        
        # Degradation Cost
        # MetaDataset: - DEGRATIO * MAXP * (Energy(>0)) -> Only discharge
        deg_cost = degradation_cost * q_rt_dis[t] * dt # Is it per MWh? Yes, DEGRATIO is $/MWh.
        # But wait, MetaDataset degradation calc: - self.DEGRATIO * self.MAXP * (energy_action_rt (>0))
        # energy_action is normalized (-1 to 1). MAXP is MW.
        # So DEGRATIO * Energy_MW. This is cost per STEP? or rate?
        # Usually DEGRATIO is $/MWh-throughput.
        # If formula is DEGRATIO * Power, and added to reward per step.
        # If reward is summed, then total cost = Sum(Deg * Power).
        # Depending on if DEGRATIO is scaled by time or not.
        # MetaDataset DEGRATIO is sampled ~50.
        # Typically $/MWh. So Cost = 50 * Power_MW * dt_h.
        # Code: reward_degradation = - self.DEGRATIO * self.MAXP * (...)
        # It creates a reward term directly.
        # If the reward accumulates to total profit, we must assume DEGRATIO in code is implicitly handled or scaled.
        # However, usually cost is Power * dt (Energy) * Price.
        # If code doesn't multiply by dt (1/12), then DEGRATIO in code might be huge or interpreted as $/MW-step?
        # Let's look closer at training code.
        # total_epoches trained.
        # In MetaDataset, it simply sums `reward_market_revenue` and `reward_degradation`.
        # `reward_market_revenue` is Power * Price * MAXP * (dt? No).
        # Wait. `reward_market_revenue_per_market = ... * self.MAXP`.
        # `rev_energy_base = settlement(...)`. Settlement is Q * P.
        # If P is $/MWh. Q is normalized [0,1].
        # Revenue = Q * P * MAXP = [1] * [$/MWh] * [MW]. Unit is $/h.
        # If we sum $/h over steps without * dt, we get something proportional to energy but scaled by 12.
        # Unless Environment step size is implicitly considered 1 unit.
        # For Optimization, to get REAL dollars, we should use * dt.
        # But for comparison with RL agent reward (which might be unscaled), we might need to match format.
        # Generally, Profit ($) = sum( Power(MW) * Price($/MWh) * dt(h) ).
        # If RL env returns Reward = Power * Price, then RL Reward is rate ($/h).
        # Total Eps Reward = Sum(Rate). Real Profit = Sum(Rate) * dt.
        # We will calculate REAL PROFIT here.
        
        # Revenue
        obj += rev_energy * dt
        
        # Cost
        obj -= deg_cost # deg_cost already includes dt
        
    m.setObjective(obj, GRB.MAXIMIZE)
    
    m.optimize()
    
    if m.status == GRB.OPTIMAL:
        print("\nOptimal Solution Found")
        total_obj = m.objVal
        print(f"Total Objective (Profit): ${total_obj:.2f}")
        
        # Extract Results
        q_da_val = np.array([q_da[t].x for t in range(T)])
        q_rt_val = np.array([q_rt[t].x for t in range(T)])
        soc_val = np.array([soc[t].x for t in range(T+1)])
        
        # Plotting
        plt.figure(figsize=(15, 10))
        
        # 1. Prices
        plt.subplot(3, 1, 1)
        plt.plot(rt_prices['energy'], label='RT Price', color='orange')
        plt.plot(da_prices['energy'], label='DA Price', color='cyan', linestyle='--')
        plt.legend()
        plt.title("Energy Prices ($/MWh)")
        plt.grid(True, alpha=0.3)
        
        # 2. Actions
        plt.subplot(3, 1, 2)
        plt.plot(q_rt_val, label='RT Dispatch (MW)', color='blue')
        plt.plot(q_da_val, label='DA Commitment (MW)', color='red', linestyle='--')
        plt.legend()
        plt.title("Optimal Dispatch")
        plt.ylabel("MW")
        plt.grid(True, alpha=0.3)
        
        # 3. SoC
        plt.subplot(3, 1, 3)
        plt.plot(soc_val, label='SoC (MWh)', color='green')
        plt.axhline(y=soc_max_mwh, color='k', linestyle=':', alpha=0.5)
        plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        plt.legend()
        plt.title("State of Charge")
        plt.xlabel("Time Step (5-min)")
        plt.ylabel("MWh")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("optimal_bidding_result.png")
        print("Plot saved to optimal_bidding_result.png")
        
        return
    else:
        print("Optimization failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="meta_bidding/data/pjm_data/pjm_price_train_dual.pkl")
    parser.add_argument("--node", default="AECO")
    parser.add_argument("--days", type=int, default=1)
    args = parser.parse_args()
    
    solve_optimal_bidding(
        price_data_path=args.data_path,
        node=args.node,
        horizon_hours=24*args.days
    )
