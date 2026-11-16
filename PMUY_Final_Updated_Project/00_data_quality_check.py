from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent
df = pd.read_excel(ROOT / "data" / "West_Bengal_PMUY_NFHS5_Combined.xlsx")
years = ["2016-17","2017-18","2018-19","2019-20","2020-21","2021-22","2022-23_Apr-Feb"]
print("Shape:", df.shape)
print("Duplicate district rows:", df["District"].duplicated().sum())
print("Missing values by column:")
print(df.isna().sum().to_string())
print("Recorded 2020-21 zeros:", int((pd.to_numeric(df["2020-21"], errors="coerce") == 0).sum()))
print("2022-23 is a partial April-February period; exclude it from complete-year model comparison.")
