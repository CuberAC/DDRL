import sys
import os
import argparse
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add root to sys.path to find meta_bidding
sys.path.append(os.getcwd())

from meta_bidding.train.ddrl.ddrl_trainer import LSTMTrainer

def plot_results(output_dir, date_str, results, num_markets):
    # results is a dict containing time-series data for ONE day (288 steps)
    # create plots
    steps = np.arange(288)
    
    # Plot 1: DA Plan vs RT Action (Energy)
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Energy is market 0
    da_action_energy = results['da_action'][:, 0] # 24 hours
    # Repeat DA action to 288 steps for plotting
    da_action_energy_long = np.repeat(da_action_energy, 12)
    rt_action_energy = results['rt_action'][:, 0]
    
    axes[0].step(steps, da_action_energy_long, label='Day-Ahead Plan (Energy)', where='post', color='blue', linestyle='--')
    axes[0].plot(steps, rt_action_energy, label='Real-Time Action (Energy)', alpha=0.7, color='red')
    axes[0].set_ylabel('Power (MW)')
    axes[0].set_title(f'Energy Bidding Strategy - {date_str}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Prices (Energy)
    rt_price = results['lmp'][:, 0]
    da_price = results['lmp_da'][:, 0]
    # DA Price is sampled per step, so it should be same shape
    axes[1].plot(steps, da_price, label='Day-Ahead Price', linestyle='--', color='blue')
    axes[1].plot(steps, rt_price, label='Real-Time Price', color='red', alpha=0.7)
    axes[1].set_ylabel('Price ($/MWh)')
    axes[1].set_title('Market Prices')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: SoC
    soc = results['soc']
    da_soc = results.get('da_soc', None)
    
    axes[2].plot(steps, soc, label='Real-Time SoC', color='green')
    if da_soc is not None:
        axes[2].plot(steps, da_soc, label='Day-Ahead Virtual SoC', color='blue', linestyle='--', alpha=0.8)
        
    axes[2].set_ylabel('SoC (0-1)')
    axes[2].set_xlabel('Time Step (5-min intervals)')
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{date_str}_analysis.png'))
    plt.close()

def evaluate_model(model_path, env_config, device):
    # Get model name for folder
    checkpoint_name = os.path.basename(model_path).replace('.pth', '')
    output_dir = os.path.join(os.path.dirname(model_path), checkpoint_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Evaluating model: {model_path}")
    print(f"Output directory: {output_dir}")
    
    # Load model
    trainer = LSTMTrainer(batch_size=1, seq_len=1, device=device, env_config=env_config)
    trainer.load(model_path)
    trainer.eval()
    
    # Get test days indices
    # We want to iterate uniquely.
    all_sod_idx = np.sort(np.unique(trainer.env.sod_idx))
    
    daily_stats = []
    
    # We should skip the first 4 days of each month-block ensuring history continuity.
    # The split logic was: Days > 23 for each month.
    # So each month starts with Day 24.
    # We need to skip 24, 25, 26, 27. Start evaluation from Day 28.
    # Let's Inspect dates to be robust.
    
    # Pre-fetch all dates for all SOD indices to filter
    valid_indices = []
    for idx in all_sod_idx:
        try:
            date_obj = trainer.env.df_data.iloc[idx]['SETTLEMENTDATE']
            # If day < 28, it means this day relies on history from previous month (Day 20-23)
            # which is missing in our Test Split (Gap).
            # So we only keep days >= 28.
            if date_obj.day >= 28:
                valid_indices.append(idx)
        except:
             continue
             
    print(f"Filtering non-continuous days (Day < 28). Remaining test days: {len(valid_indices)} out of {len(all_sod_idx)}")

    for idx in valid_indices:
        # Reset environment
        trainer.env.reset()
        
        # KEY: Force specific day and initial SoC
        trainer.env._pcs = np.array([idx])
        # Start each test day with 50% SoC for consistent comparison
        trainer.env._soc = [torch.tensor([0.5], device=device)] 
        
        # Get Date string
        try:
            date_obj = trainer.env.df_data.iloc[idx]['SETTLEMENTDATE']
            date_str = pd.to_datetime(date_obj).strftime('%Y-%m-%d')
        except:
             print(f"Skipping index {idx}, cannot retrieve date.")
             continue
        
        # Run one day (24 hours = 288 steps)
        with torch.no_grad():
             info = trainer.one_minibatch_step(verbose=True)
        
        # Process Info
        # DA Revenue (Sum over day)
        # Note: info returns instantaneous revenue rate ($/h) per step (5min).
        # We must divide by 12 to get actual USD ($).
        da_rev_sum = np.sum(info['rev_da']) / 12.0
        total_rev_sum = np.sum(info['rev_total']) / 12.0
        # Use provided total/da to compute RT component
        rt_rev_sum = total_rev_sum - da_rev_sum 
        
        daily_stats.append({
            'Date': date_str,
            'Day-Ahead Revenue ($)': da_rev_sum,
            'Real-Time Revenue ($)': rt_rev_sum,
            'Total Revenue ($)': total_rev_sum
        })
        
        # Format for plotting
        # info['da_action'] shape is (Batch=1, Markets, 24) -> Need (24, Markets)
        # info['action'] shape is (Steps=288, Batch=1, Markets) -> Need (288, Markets)
        # info['lmp'] shape is (Steps=288, Batch=1, Markets) -> Need (288, Markets)
        
        # Calculate Day-Ahead Virtual SoC
        eff = trainer.env.EFFICIENCY.item() # Scalar sqrt(0.9)
        max_ptr_ratio = trainer.env.MAXPRTRATIO.item() # Scalar
        
        # Check if tensor or numpy
        da_action_container = info['da_action'][0][0]
        if isinstance(da_action_container, torch.Tensor):
             da_action_energy = da_action_container.detach().cpu().numpy()
        else:
             da_action_energy = da_action_container
             
        da_soc_hourly = [0.5] # Start at 0.5
        current_soc = 0.5
        
        for h in range(24):
            power = da_action_energy[h]
            # If power > 0 (Discharge): SoC decreases by Power * Rate / Efficiency
            # If power < 0 (Charge): SoC increases by |Power| * Rate * Efficiency -> SoC decreases by Power * Rate * Efficiency
            # Wait, Charge increases SoC.
            # Delta = - Power * Rate * Eff (if P<0)
            # Delta = - Power * Rate / Eff (if P>0)
            
            # Hourly Rate = max_ptr_ratio * 12
            rate = max_ptr_ratio * 12.0
            
            if power > 0:
                delta = power * rate / eff
            else:
                delta = power * rate * eff
                
            current_soc = current_soc - delta
            # Clip to physical limits for realistic "virtual" trace
            current_soc = np.clip(current_soc, 0.0, 1.0)
            da_soc_hourly.append(current_soc)
            
        # Expand da_soc_hourly (25 points) to 288 points?
        # Step plot style: SOC is state at BEGINNING of hour? 
        # Usually SoC is plotted as continuous state. Initial SoC is at t=0. 
        # After hour 1 (12 steps), SoC is da_soc_hourly[1].
        # So we interpolate or step.
        # Let's create a 288-length array doing linear interpolation for nicer visualization 
        # OR step if we treat it as discrete blocks.
        # Real-time SoC updates every 5 mins. DA is hourly constant power -> Linear SoC change.
        
        da_soc_series = []
        for h in range(24):
            start_s = da_soc_hourly[h]
            end_s = da_soc_hourly[h+1]
            # Linear interp for 12 steps
            segment = np.linspace(start_s, end_s, 12, endpoint=False) # Endpoint False to align with next start?
            # Actually, standard is to include endpoints?
            # Let's use linspace(start, end, 13) and drop last?
            segment = np.linspace(start_s, end_s, 13)[:-1]
            da_soc_series.extend(segment)
        da_soc_series = np.array(da_soc_series)

        plot_data = {
            'da_action': info['da_action'][0].T, # Transpose to (24, M)
            'rt_action': info['action'][:, 0, :], 
            'lmp': info['lmp'][:, 0, :],
            'lmp_da': info['lmp_da'][:, 0, :],
            'soc': info['soc'][:, 0],
            'da_soc': da_soc_series
        }
        
        plot_results(output_dir, date_str, plot_data, trainer.num_markets)
        
    # Save Summary CSV
    df_res = pd.DataFrame(daily_stats)
    
    # [NEW] Add Optimality Ratio
    optimal_path = "/root/DDRL/optimal_bidding_year_results.csv"
    if os.path.exists(optimal_path):
        try:
             df_optimal = pd.read_csv(optimal_path)
             # df_optimal uses 'date', we use 'Date'. 'date' format matches 'YYYY-MM-DD' if consistent
             df_optimal['date'] = pd.to_datetime(df_optimal['date']).dt.strftime('%Y-%m-%d')
             optimal_map = dict(zip(df_optimal['date'], df_optimal['net_profit']))
             
             def get_ratio(row):
                 d = row['Date']
                 if d in optimal_map and optimal_map[d] != 0:
                     return (row['Total Revenue ($)'] / optimal_map[d]) * 100.0
                 return np.nan
                 
             df_res['Percent of Optimal (%)'] = df_res.apply(get_ratio, axis=1)
        except Exception as e:
             print(f"Warning: Failed to load optimal results: {e}")
    
    # Calculate averages
    avg_data = {
        'Date': 'Average',
        'Day-Ahead Revenue ($)': df_res['Day-Ahead Revenue ($)'].mean(),
        'Real-Time Revenue ($)': df_res['Real-Time Revenue ($)'].mean(),
        'Total Revenue ($)': df_res['Total Revenue ($)'].mean()
    }
    
    if 'Percent of Optimal (%)' in df_res.columns:
        avg_data['Percent of Optimal (%)'] = df_res['Percent of Optimal (%)'].mean()
        
    avg_row = pd.DataFrame([avg_data])
    df_res = pd.concat([df_res, avg_row], ignore_index=True)
    
    csv_path = os.path.join(output_dir, 'daily_performance_summary.csv')
    df_res.to_csv(csv_path, index=False)
    print(f"Saved evaluation results to {csv_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing .pth files")
    parser.add_argument("--node", type=str, default="AECO")
    parser.add_argument("--product", nargs='+', default=['energy'], help="Markets to participate in")
    parser.add_argument("--degradation_cost", type=float, default=10.0)
    parser.add_argument("--soc", type=float, default=4.0, help="Battery duration in hours")
    args = parser.parse_args()
    
    # Calculate num_markets matching MetaDataset logic
    num_markets = 0
    if 'energy' in args.product: num_markets += 1
    if 'regulation' in args.product: num_markets += 2
    if 'reserve' in args.product: num_markets += 6

    # Configuration matching training
    env_config = {
        'product': args.product, 
        'num_markets': num_markets, # Explicitly set num_markets
        'soc': args.soc, 
        'node': [args.node],
        'data_source': 'test', # Use test set
        'degradation_cost': args.degradation_cost,
        'device': 0 # Force CUDA:0
    }
    
    print(f"Configuration: {env_config}")

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Find all .pth files in the directory
    # Sort numerically by filename (0.pth, 50.pth, ...)
    pth_files = glob.glob(os.path.join(args.model_dir, "*.pth"))
    
    if not pth_files:
        print(f"No .pth files found in {args.model_dir}")
        return

    # Sort key function: extract number from filename
    def get_epoch_num(filepath):
        base = os.path.basename(filepath)
        name, _ = os.path.splitext(base)
        try:
            return int(name)
        except ValueError:
            return -1 # For non-numeric names
            
    checkpoints = sorted(pth_files, key=get_epoch_num)
    
    print(f"Found {len(checkpoints)} checkpoints to evaluate.")
    
    for ckpt in checkpoints:
        if get_epoch_num(ckpt) < 0:
            print(f"Skipping non-epoch checkpoint: {ckpt}")
            continue
        # Use a fresh environment for each checkpoint or handle data indices carefully
        # Actually evaluate_model re-initializes LSTMTrainer which re-initializes Environment.
        # So we can safely loop.
        evaluate_model(ckpt, env_config, device)

if __name__ == "__main__":
    main()
