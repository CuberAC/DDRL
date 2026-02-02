import pandas as pd
import glob
import os
import argparse

def update_tables_with_optimality(top_model_dir):
    # 1. Load Optimal Bidding Results
    optimal_path = "/root/DDRL/optimal_bidding_year_results.csv"
    if not os.path.exists(optimal_path):
        print(f"Error: Optimal results file not found at {optimal_path}")
        return

    df_optimal = pd.read_csv(optimal_path)
    # Ensure consistent date format
    df_optimal['date'] = pd.to_datetime(df_optimal['date']).dt.strftime('%Y-%m-%d')
    # Create lookup map for net_profit
    optimal_map = dict(zip(df_optimal['date'], df_optimal['net_profit']))

    print(f"Loaded {len(optimal_map)} optimal records.")

    # 2. Find all daily_performance_summary.csv files recursively
    # Structure is model_dir/checkbox_index/daily_performance_summary.csv
    # Or just search recursively
    search_pattern = os.path.join(top_model_dir, "**", "daily_performance_summary.csv")
    csv_files = glob.glob(search_pattern, recursive=True)

    print(f"Found {len(csv_files)} summary files to update.")

    for csv_file in csv_files:
        print(f"Updating {csv_file}...")
        try:
            df = pd.read_csv(csv_file)
            
            # Check if columns needed are there
            if 'Total Revenue ($)' not in df.columns or 'Date' not in df.columns:
                print(f"Skipping {csv_file}: Missing required columns.")
                continue

            # Calculate Percentage
            # We want to add column "Optimality Gap (%)" or "Percent of Optimal (%)"
            # Logic: row['Total Revenue ($)'] / optimal_map[row['Date']] * 100
            
            def get_optimal_ratio(row):
                date_str = row['Date']
                if date_str == 'Average':
                    return None # Will recalc average later
                
                if date_str in optimal_map:
                    opt_profit = optimal_map[date_str]
                    current_profit = row['Total Revenue ($)']
                    
                    if opt_profit == 0:
                        return 0.0 # Avoid division by zero
                    
                    # If opt profit is negative? Usually it's positive.
                    return (current_profit / opt_profit) * 100.0
                else:
                    return None

            df['Percent of Optimal (%)'] = df.apply(get_optimal_ratio, axis=1)

            # Re-calculate Average row
            # Filter out the existing Average row to recalc
            df_data = df[df['Date'] != 'Average'].copy()
            
            # Calculate mean of percentage
            avg_percent = df_data['Percent of Optimal (%)'].mean()
            
            # Locate the Average row index in original df
            avg_row_idx = df.index[df['Date'] == 'Average'].tolist()
            
            if avg_row_idx:
                idx = avg_row_idx[0]
                df.at[idx, 'Percent of Optimal (%)'] = avg_percent
            else:
                 # If Average row didn't exist for some reason, append it?
                 # Assuming it exists based on previous code.
                 pass

            # Save back
            df.to_csv(csv_file, index=False)
            
        except Exception as e:
            print(f"Failed to process {csv_file}: {e}")

    print("Update complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Root directory searching containing summary csvs")
    args = parser.parse_args()
    
    update_tables_with_optimality(args.dir)
