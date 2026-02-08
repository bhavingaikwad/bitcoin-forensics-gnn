import pandas as pd
import os

# Create a sample directory that isn't ignored by git
os.makedirs('data_sample', exist_ok=True)

# Save small snippets (first 1000 rows)
pd.read_csv('data/elliptic_txs_classes.csv').head(1000).to_csv('data_sample/elliptic_txs_classes.csv', index=False)
pd.read_csv('data/elliptic_txs_features.csv', header=None).head(1000).to_csv('data_sample/elliptic_txs_features.csv', index=False, header=False)

print("✅ Forensic samples created in 'data_sample/'")