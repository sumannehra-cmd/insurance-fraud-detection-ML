import pandas as pd

df = pd.read_csv('data/fraud_oracle.csv')
print("Column names in dataset:")
print(df.columns.tolist())
print("\nFirst 2 rows:")
print(df.head(2))