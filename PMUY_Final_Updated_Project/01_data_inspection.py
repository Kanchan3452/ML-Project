from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'West_Bengal_PMUY_NFHS5_Combined.xlsx'
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)

import pandas as pd

# Load dataset
df = pd.read_excel(DATA)

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Information:")
print(df.info())

print("\nNumber of Districts:", df["District"].nunique())
