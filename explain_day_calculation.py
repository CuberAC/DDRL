import sys
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Add root to sys.path to find meta_bidding
sys.path.append(os.getcwd())

# Import necessary classes
from meta_bidding.train.ddrl.ddrl_trainer import LSTMTrainer

def explicit_calculation_check(model_path, target_date_str='2022-01-30', env_config=None, device='cuda:0'):
    print(f"Loading model from {model_path}...")
    trainer = LSTMTrainer(batch_size=1, seq_len=1, device=device, env_config=env_config)
    trainer.load(model_path)
    trainer.eval()
    
    # 1. Find the index correspond to the target date
    target_idx = -1
    for idx_candidate in np.unique(trainer.env.sod_idx):
        try:
            date_obj = trainer.env.df_data.iloc[idx_candidate]['SETTLEMENTDATE']
            date_str = pd.to_datetime(date_obj).strftime('%Y-%m-%d')
            if date_str == target_date_str:
                target_idx = idx_candidate
                break
        except:
            continue
            
    if target_idx == -1:
        print(f"Error: Date {target_date_str} not found in test dataset.")
        return

    print(f"Found {target_date_str} at index {target_idx}")
    
    # 2. Reset Env to that day
    trainer.env.reset()
    trainer.env._pcs = np.array([target_idx])
    # Force SoC to 0.5 as per evaluation script
    trainer.env._soc = [torch.tensor([0.5], device=device)] 
    
    # 3. Run one day
    with torch.no_grad():
        info = trainer.one_minibatch_step(verbose=True)
    
    # 4. Extract Data
    # Shape Checks:
    # lmp: (288, Batch, Markets)
    # lmp_da: (24, Batch, Markets) -> Actually lmp_da in info might be 288 if expanded?
    # Let's inspect info['lmp_da']. If it's 288, it's repeated.
    
    # Handle both Tensor and Numpy
    def to_np(x):
        if hasattr(x, 'cpu'):
            return x.detach().cpu().numpy()
        return x
        
    lmp_rt_288 = to_np(info['lmp'])[:, 0, 0] # 288 steps (Energy Market 0)
    
    lmp_da_raw = to_np(info['lmp_da'])
    if lmp_da_raw.shape[0] == 288:
        # Take every 12th element to get hourly DA prices
        lmp_da_24 = lmp_da_raw[::12, 0, 0]
    else:
        lmp_da_24 = lmp_da_raw[:, 0, 0]
        
    da_action_24 = to_np(info['da_action'][0][0]) # Shape (24,)
    rt_action_288 = to_np(info['action'])[:, 0, 0] # Shape (288,)
    
    # 5. Perform Explicit Calculation
    print("\n" + "="*80)
    print(f"       REVENUE CALCULATION BREAKDOWN FOR {target_date_str} (ENERGY MARKET)")
    print("="*80 + "\n")
    
    print(f"{'Hour':<5} | {'DA Price ($/MWh)':<15} | {'DA Plan (MW)':<15} | {'DA Revenue ($)':<15} || {'RT Rev Sum ($)':<15} | {'Total ($)':<15}")
    print("-" * 90)
    
    total_rev_da_acc = 0.0
    total_rev_rt_acc = 0.0
    total_rev_final = 0.0
    
    detailed_log = []
    
    for h in range(24):
        # DA Part
        p_da = lmp_da_24[h]
        q_da = da_action_24[h]
        
        # DA Revenue for this hour = Price * Quantity
        # Assumption: P is $/MWh, Q is MW. Duration is 1 Hour.
        # So Revenue = P * Q * 1h
        da_rev_h = p_da * q_da
        total_rev_da_acc += da_rev_h
        
        # RT Part
        # Sum over 12 intervals for this hour
        rt_rev_h_acc = 0.0
        
        start_idx = h * 12
        end_idx = (h + 1) * 12
        
        rt_prices_h = lmp_rt_288[start_idx:end_idx]
        rt_actions_h = rt_action_288[start_idx:end_idx]
        
        step_details = []
        for t in range(12):
            p_rt = rt_prices_h[t]
            q_rt = rt_actions_h[t]
            
            # Settlement Rule:
            # Rev = Q_da * P_da + (Q_rt - Q_da) * P_rt
            # BUT wait, the environment calculates this every 5 mins.
            # Does Env divide by 12?
            # User CSV: 2022-01-30 Total DA Rev ~ -3890.
            # Let's see if Sum(P_da * Q_da) matches that.
            
            # Settlement Logic verification:
            # deviation = (q_rt - q_da)
            # rt_component = deviation * p_rt
            
            # Question: Does Env return accumulated DA rev per step?
            # In MetaDataset: rev_energy_base = q_da * p_da + (q_rt - q_da) * p_rt
            # This is instantaneous rate ($/h) * 1 ?? No, P is $/MWh.
            # If Env does NOT divide by 12, then Sum(rev) over 12 steps is 12 * Actual $.
            # Or maybe P_da is handled differently.
            
            # Let's calculate purely as 'Rate' first
            rate_rt_component = (q_rt - q_da) * p_rt
            
            # Assume we sum these 'rates'? Or divide by 12?
            # I will print the raw sum and see if it makes sense with the total.
            
            rt_rev_h_acc += rate_rt_component
            
            step_details.append(f"    t={t+1:02d}: RT_P={p_rt:6.2f}, RT_Q={q_rt:6.2f}, Dev={q_rt-q_da:6.2f} -> RT_Comp_Rate={rate_rt_component:8.2f}")

        # Store for printing
        detailed_log.append({
            'hour': h,
            'p_da': p_da,
            'q_da': q_da,
            'da_rev_h': da_rev_h,
            'rt_rev_h_sum_rates': rt_rev_h_acc,
            'step_logs': step_details
        })
        
        total_rev_rt_acc += rt_rev_h_acc

    # Post-Calculation Analysis
    
    # -----------------------------------------------------
    # TABLE 1: Day-Ahead Financial Position
    # -----------------------------------------------------
    print("\n" + "="*90)
    print("                 TABLE 1: DAY-AHEAD FINANCIAL POSITION (24 Hours)")
    print("="*90)
    print(f"{'Hour':<6} | {'DA Price ($/MWh)':<18} | {'DA Plan (MW)':<15} | {'DA Revenue ($)':<15} | {'Note'}")
    print("-" * 90)
    
    total_da_rev_usd = 0.0
    for h in range(24):
        p = lmp_da_24[h]
        q = da_action_24[h]
        rev = p * q # 1 Hour
        total_da_rev_usd += rev
        note = "Charge" if q < 0 else "Discharge" if q > 0 else "Idle"
        print(f"{h:<6} | {p:<18.2f} | {q:<15.2f} | {rev:<15.2f} | {note}")
        
    print("-" * 90)
    print(f"{'TOTAL':<6} | {'':<18} | {'':<15} | {total_da_rev_usd:<15.2f} | Based on Hourly P*Q")


    # -----------------------------------------------------
    # TABLE 2: Real-Time Settlement (288 Intervals)
    # -----------------------------------------------------
    print("\n" + "="*110)
    print("                 TABLE 2: REAL-TIME DEVIATION SETTLEMENT (288 Intervals)")
    print("="*110)
    print(f"{'Time':<8} | {'RT Price':<10} | {'RT Act':<8} | {'DA Commit':<10} | {'Dev(MW)':<10} | {'Rate($/h)':<10} | {'StepRev($)':<12}")
    print(f"{'(h:m)':<8} | {'($/MWh)':<10} | {'(MW)':<8} | {'(MW)':<10} | {'(RT-DA)':<10} | {'(Dev*P)':<10} | {'(Rate/12)':<12}")
    print("-" * 110)
    
    total_rt_rev_usd = 0.0
    
    rt_table_data = [] # Just for potential file writing later or debugging
    
    for h in range(24):
        # Retrieve DA commitment for this hour
        q_da_h = da_action_24[h]
        
        # Loop over 12 intervals (5 mins each)
        for t in range(12):
            global_step = h * 12 + t
            p_rt = lmp_rt_288[global_step]
            q_rt = rt_action_288[global_step]
            
            deviation = q_rt - q_da_h
            settlement_rate = deviation * p_rt # $/h
            step_revenue = settlement_rate / 12.0 # Actual $ for 5 mins
            
            total_rt_rev_usd += step_revenue
            
            time_str = f"{h:02d}:{t*5:02d}"
            
            print(f"{time_str:<8} | {p_rt:<10.2f} | {q_rt:<8.2f} | {q_da_h:<10.2f} | {deviation:<10.2f} | {settlement_rate:<10.2f} | {step_revenue:<12.2f}")

    print("-" * 110)
    print(f"{'TOTAL':<8} | {'':<10} | {'':<8} | {'':<10} | {'':<10} | {'':<10} | {total_rt_rev_usd:<12.2f}")
    
    print("\n" + "="*80)
    print("FINAL DAILY SUMMARY")
    print("="*80)
    print(f"Day-Ahead Revenue (Financial) : ${total_da_rev_usd:.2f}")
    print(f"Real-Time Deviation Revenue   : ${total_rt_rev_usd:.2f}")
    print(f"Total Daily Profit            : ${total_da_rev_usd + total_rt_rev_usd:.2f}")
    print("="*80)


if __name__ == "__main__":
    # Config matching run
    config = {
        'product': ['energy'], 
        'num_markets': 1, 
        'soc': 4.0, 
        'node': ['AECO'],
        'data_source': 'test', 
        'degradation_cost': 10.0,
        'device': 0 
    }
    
    model_path = "/root/DDRL/logs/metabidding-ddrl/meta-ddrl-default/20260202-0837/1950.pth"
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        
    explicit_calculation_check(model_path, env_config=config)
