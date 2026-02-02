import pandas as pd

path = "meta_bidding/data/pjm_data/pjm_price_train_dual.pkl"
try:
    df = pd.read_pickle(path)
    print(f"Loaded {path}")
    print(f"Columns: {df.columns}")
    
    if 'RRP' in df.columns:
        print("\nStatistics for RRP (RT Price):")
        print(df['RRP'].describe())
    
    if 'DA_RRP' in df.columns:
        print("\nStatistics for DA_RRP (DA Price):")
        print(df['DA_RRP'].describe())

    if 'REGIONID' in df.columns:
        print(f"\nUnique REGIONIDs: {df['REGIONID'].unique()}")
        
    # Check AECO node specifically
    df_aeco = df[df['REGIONID'] == 'AECO']
    print("\nAESO Node Stats:")
    print(df_aeco[['RRP', 'DA_RRP']].describe())
    
except Exception as e:
    print(e)
