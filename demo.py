# import pandas as pd
# import kagglehub
# from kagglehub import KaggleDatasetAdapter

# file_path = "PS_20174392719_1491204439457_log.csv"

# print("Loading dataset via kagglehub...")
# df = kagglehub.dataset_load(
#     KaggleDatasetAdapter.PANDAS,
#     "ealaxi/paysim1",
#     file_path,
# )
# print(f"Original dataset shape: {df.shape}")

# # Step 1: Keep only fraud-relevant transaction types (Reduces to ~2.7M rows)
# df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])]
# print(f"Shape after filtering TRANSFER and CASH_OUT: {df.shape}")

# # Step 2: Extract ALL fraud cases (~8,213 fraud rows)
# fraud_rows = df[df['isFraud'] == 1]
# num_fraud = len(fraud_rows)

# # Step 3: Sample clean data for graph construction
# # We want 500,000 cases in total for our final dataset demo
# target_total_rows = 500_000
# num_clean_needed = target_total_rows - num_fraud

# df_clean = df[df['isFraud'] == 0].sample(n=num_clean_needed, random_state=42)

# # Step 4: Concatenate to get our final mixed dataset
# df_final = pd.concat([fraud_rows, df_clean])

# # Optional but recommended: Shuffle dataset so frauds aren't grouped at the start/end
# df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
# print(f"Final sampled dataset shape: {df_final.shape} (Fraud Rows: {num_fraud})")

# # === BEST METHOD TO SAVE AND LOAD ===
# # Parquet is fundamentally superior to CSV for fast data queries and manipulation with Pandas.
# output_file = 'sampled_transactions.parquet'
# df_final.to_parquet(output_file, engine='pyarrow', index=False)

# print(f"\nSuccessfully stored in highly optimized Parquet format: {output_file}")
# print(f"To load it ultra-fast later, just run:")
# print(f"    df = pd.read_parquet('{output_file}')")


import pandas as pd
from data_ingestion import ingest_data_from_parquet

# This will load in a fraction of a second!
df = ingest_data_from_parquet('sampled_transactions.parquet')

print(f"Loaded {len(df)} rows ready for graph construction.")
