import pandas as pd
import numpy as np
import os

def split_dataset(
    original_file_path="meta_bidding/data/pjm_data/pjm_price_train_dual.pkl",
    train_output_path="meta_bidding/data/pjm_data/pjm_price_train_dual_split.pkl",
    test_output_path="meta_bidding/data/pjm_data/pjm_price_test_dual_split.pkl",
    split_day=23
):
    """
    Split the dataset into training and testing sets based on the day of the month.
    Train: Days <= split_day (e.g., 1-23)
    Test: Days > split_day (e.g., 24-31)
    """
    
    print(f"Loading original data from {original_file_path}...")
    try:
        df = pd.read_pickle(original_file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Check if SETTLEMENTDATE exists
    if 'SETTLEMENTDATE' not in df.columns:
        print("Error: 'SETTLEMENTDATE' column not found in dataset.")
        return
        
    print(f"Total rows: {len(df)}")
    
    # Create masks based on day of month
    # We need to ensure dt accessor works
    if not pd.api.types.is_datetime64_any_dtype(df['SETTLEMENTDATE']):
        df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])

    # Filtering logic
    # Train set: Day of month <= 23
    train_mask = df['SETTLEMENTDATE'].dt.day <= split_day
    
    # Test set: Day of month > 23
    test_mask = df['SETTLEMENTDATE'].dt.day > split_day
    
    # Split
    df_train = df[train_mask].copy().reset_index(drop=True)
    df_test = df[test_mask].copy().reset_index(drop=True)
    
    print(f"\n[Splitting Config] Cutoff Day: {split_day}")
    print(f"Training set size: {len(df_train)} rows ({len(df_train)/len(df)*100:.1f}%)")
    print(f"Testing set size:  {len(df_test)} rows ({len(df_test)/len(df)*100:.1f}%)")
    
    # Save
    print(f"\nSaving training set to {train_output_path}...")
    df_train.to_pickle(train_output_path)
    
    print(f"Saving testing set to {test_output_path}...")
    df_test.to_pickle(test_output_path)
    
    print("\nDone. Dataset split complete.")

if __name__ == "__main__":
    split_dataset()
