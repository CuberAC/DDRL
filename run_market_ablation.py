import sys
import os
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Add root to sys.path
sys.path.append(os.getcwd())

from meta_bidding.train.ddrl.ddrl_trainer import LSTMTrainer

def run_market_ablation(model_path, output_csv="market_ablation_results.csv", device='cuda:0', env_config=None):
    print(f"Loading model from {model_path}...")
    
    # Init Trainer
    trainer = LSTMTrainer(batch_size=1, seq_len=1, device=device, env_config=env_config)
    trainer.load(model_path)
    trainer.eval()
    
    # Get Test Dates (Days >= 28)
    test_indices = []
    
    # Use existing helper or logic to find indices
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
    
    results = []
    
    # Loop over test days
    for idx, date_str in tqdm(test_indices):
        # We need to run the model ONCE to get the actions (DA and RT policies)
        # Then manually calculate revenues for 3 scenarios:
        # 1. Dual Market (Standard)
        # 2. DA Only (Force Q_rt = Q_da)
        # 3. RT Only (Force Q_da = 0)
        
        # Reset Env
        trainer.env.reset()
        trainer.env._pcs = np.array([idx])
        trainer.env._soc = [torch.tensor([0.5], device=device)]
        
        # Run one day
        with torch.no_grad():
            info = trainer.one_minibatch_step(verbose=True)
            
        # Extract Data
        # Shapes:
        # lmp: (288, 1, 9) -> RT Prices
        # lmp_da: (24, 1, 9) -> DA Prices (Needs expansion or careful indexing)
        # action: (288, 1, 9) -> Q_rt
        # da_action: (1, 9, 24) -> Q_da (Needs validation of shape from previous debug)
        
        # Check shapes from info
        # info['da_action'] is (Batch, Markets, 24) -> (1, 9, 24)
        # We need (24, 9) for calculation
        q_da_raw = info['da_action'][0].T # (24, 9)
        
        # info['action'] is (Steps, Batch, Markets) -> (288, 1, 9)
        q_rt_raw = info['action'][:, 0, :] # (288, 9)
        
        # Prices
        p_rt_raw = info['lmp'][:, 0, :] # (288, 9)
        p_da_raw_short = info['lmp_da'][:, 0, :] # (24, 9) usually, check definition in trainer
        # In trainer: lmps_da.append(info['lmp_da']) -> stacked 24 times.
        # So p_da_raw_short is (24, 9).
        
        # We only care about Energy Market (Index 0) for this analysis
        q_da_24 = q_da_raw[:, 0]
        q_rt_288 = q_rt_raw[:, 0]
        p_rt_288 = p_rt_raw[:, 0]
        p_da_24 = p_da_raw_short[:, 0]
        
        # --- SCENARIO 1: DUAL MARKET (Baseline) ---
        # Rev = Sum( Q_da * (P_da - P_rt) + Q_rt * P_rt )
        # Note: DA part is hourly, RT part is 5-min.
        # P_da is hourly price. P_rt is 5-min price.
        # We need to align time resolution.
        
        rev_dual = 0.0
        rev_da_only = 0.0
        rev_rt_only = 0.0
        
        for h in range(24):
            # Hour h parameters
            q_da_h = q_da_24[h]
            p_da_h = p_da_24[h]
            
            # --- DA Only Calculation (Scenario 2) ---
            # Assumption: Perfect adherence means Q_rt = Q_da for all 12 intervals
            # Revenue = Q_da * P_da * 1 hour
            rev_da_only += q_da_h * p_da_h
            
            # RT Interval Loop
            start = h * 12
            end = start + 12
            
            p_rt_vals = p_rt_288[start:end]
            q_rt_vals = q_rt_288[start:end]
            
            for t in range(12):
                p_rt = p_rt_vals[t]
                q_rt = q_rt_vals[t]
                dt = 1.0/12.0 # 5 mins
                
                # --- RT Only Calculation (Scenario 3) ---
                # Q_da = 0. Rev = Q_rt * P_rt * dt
                # Using the Model's RT policy (which might be optimized for Dual, but we use it as proxy)
                # Or should we re-run model with da=0? 
                # Request says "use trained model... separate...". 
                # Usually implies "what if we only settled on this market".
                # If we used the model's Q_rt (which was conditioned on Q_da), implies we trust that Q_rt.
                # But strictly "RT Market Capabilty" might mean "best Q_rt given no DA".
                # Here we assume: "What is the value of the Q_rt stream if DA was zero?"
                rev_rt_only += q_rt * p_rt * dt
                
                # --- Dual Calculation (Scenario 1 check) ---
                # Rev = Q_da * (P_da - P_rt) + Q_rt * P_rt
                # Wait, P_da is hourly. P_rt varies.
                # Arbitrage term accumulation:
                # Term 1: Q_da * P_da * dt (Accumulates to Q_da * P_da * 1)
                # Term 2: -Q_da * P_rt * dt
                # Term 3: Q_rt * P_rt * dt
                
                # So step revenue = (Q_da * P_da * dt) - (Q_da * P_rt * dt) + (Q_rt * P_rt * dt)
                #                 = Q_da * (P_da - P_rt) * dt + Q_rt * P_rt * dt
                
                step_rev_dual = q_da_h * (p_da_h - p_rt) * dt + q_rt * p_rt * dt
                rev_dual += step_rev_dual

        results.append({
            'Date': date_str,
            'Dual_Market_Rev': rev_dual,
            'DA_Only_Rev': rev_da_only,
            'RT_Only_Rev': rev_rt_only
        })
        
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Add Average
    avg_row = df.mean(numeric_only=True)
    avg_row['Date'] = 'Average'
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    
    # Save
    df.to_csv(output_csv, index=False)
    print(f"Saved market ablation results to {output_csv}")
    print("Average Results:")
    print(avg_row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="market_ablation_results.csv")
    args = parser.parse_args()
    
    # Config
    env_config = {
        'product': ['energy'], 
        'num_markets': 1, 
        'soc': 4.0, 
        'node': ['AECO'],
        'data_source': 'test', 
        'degradation_cost': 10.0,
        'device': 0 
    }
    
    run_market_ablation(args.model_path, args.output, env_config=env_config)
