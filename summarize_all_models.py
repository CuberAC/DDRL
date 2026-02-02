import pandas as pd
import glob
import os
import argparse
import re

def summarize_models(model_dir):
    # Search for all daily_performance_summary.csv files
    search_pattern = os.path.join(model_dir, "**", "daily_performance_summary.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(csv_files)} summary files.")
    
    summary_data = []
    
    for csv_file in csv_files:
        try:
            # Extract checkpoint/model ID from directory structure
            # Assuming structure: .../checkpoint_id/daily_performance_summary.csv
            parent_dir = os.path.dirname(csv_file)
            folder_name = os.path.basename(parent_dir)
            
            # Try to interpret folder name as a number (checkpoint)
            # If not a number, keep as string
            try:
                checkpoint = int(folder_name)
            except ValueError:
                checkpoint = folder_name
                
            df = pd.read_csv(csv_file)
            
            # Find the 'Average' row
            avg_row = df[df['Date'] == 'Average']
            
            if not avg_row.empty:
                # Extract values
                # Date,Day-Ahead Revenue ($),Real-Time Revenue ($),Total Revenue ($),Percent of Optimal (%)
                
                # Check column existence to be safe
                da_rev = avg_row['Day-Ahead Revenue ($)'].values[0] if 'Day-Ahead Revenue ($)' in df.columns else 0
                rt_rev = avg_row['Real-Time Revenue ($)'].values[0] if 'Real-Time Revenue ($)' in df.columns else 0
                total_rev = avg_row['Total Revenue ($)'].values[0] if 'Total Revenue ($)' in df.columns else 0
                
                # Percent of Optimal might be missing if older files (though we just updated them)
                opt_pct = avg_row['Percent of Optimal (%)'].values[0] if 'Percent of Optimal (%)' in df.columns else None
                
                summary_data.append({
                    'Checkpoint': checkpoint,
                    'Total Revenue ($)': total_rev,
                    'Day-Ahead Revenue ($)': da_rev,
                    'Real-Time Revenue ($)': rt_rev,
                    'Percent of Optimal (%)': opt_pct
                })
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    if not summary_data:
        print("No data found.")
        return

    # Create DataFrame
    df_summary = pd.DataFrame(summary_data)
    
    # Sort by Checkpoint if it's numeric
    try:
        df_summary = df_summary.sort_values(by='Checkpoint')
    except:
        df_summary = df_summary.sort_values(by='Checkpoint', key=lambda col: col.astype(str))
        
    # Save to CSV
    output_path = os.path.join(model_dir, "all_models_summary.csv")
    df_summary.to_csv(output_path, index=False)
    
    print(f"\nSummary saved to: {output_path}")
    
    # Also print the best model by Total Revenue
    print("\nBest Model by Total Revenue ($):")
    best_rev = df_summary.loc[df_summary['Total Revenue ($)'].idxmax()]
    print(best_rev)
    
    # Also print the best model by Optimality Percentage if available
    if 'Percent of Optimal (%)' in df_summary.columns and not df_summary['Percent of Optimal (%)'].isnull().all():
        print("\nBest Model by Percent of Optimal (%):")
        best_opt = df_summary.loc[df_summary['Percent of Optimal (%)'].idxmax()]
        print(best_opt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Directory containing model checkpoints")
    args = parser.parse_args()
    
    summarize_models(args.dir)
