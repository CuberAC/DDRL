import pandas as pd
import pickle
import os

file_path = "meta_bidding/data/pjm_data/pjm_price_train_dual.pkl"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

try:
    print(f"Loading {file_path} using pandas...")
    df = pd.read_pickle(file_path)
    print("\n--- Data Loaded ---")
    print(f"Type: {type(df)}")
    
    if isinstance(df, pd.DataFrame):
        print(f"\nShape: {df.shape}")
        print("\nColumns:")
        for col in df.columns:
            print(f"  - {col}")
        
        print("\nHead (first 5 rows):")
        print(df.head())
        
    else:
        print("\nContent representation:")
        print(df)

except Exception as e:
    print(f"Error loading with pandas: {e}")
