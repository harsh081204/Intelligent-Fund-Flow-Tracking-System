import pandas as pd

def ingest_data_from_csv(filepath: str) -> pd.DataFrame:
    """
    Production-ready ingestion layer.
    Can be easily swapped to ingest from a DB, Kafka stream, or API feed.
    """
    print(f"Ingesting data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Standardize column types immediately
    df['step']     = df['step'].astype(int)
    df['amount']   = df['amount'].astype(float)
    df['nameOrig'] = df['nameOrig'].astype(str)
    df['nameDest'] = df['nameDest'].astype(str)
    df['isFraud']  = df['isFraud'].astype(int)
    
    # Filter to fraud-relevant types only
    df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])]
    
    # Sort by time — always
    df = df.sort_values('step').reset_index(drop=True)
    
    return df

def ingest_data_from_parquet(filepath: str) -> pd.DataFrame:
    """
    Helper function for hackathon ultra-fast loads.
    """
    print(f"Ingesting data from parquet {filepath}...")
    df = pd.read_parquet(filepath)
    
    # Standardize column types immediately
    df['step']     = df['step'].astype(int)
    df['amount']   = df['amount'].astype(float)
    df['nameOrig'] = df['nameOrig'].astype(str)
    df['nameDest'] = df['nameDest'].astype(str)
    df['isFraud']  = df['isFraud'].astype(int)
    
    # Filter to fraud-relevant types only
    df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])]
    
    # Sort by time — always
    df = df.sort_values('step').reset_index(drop=True)
    
    return df
