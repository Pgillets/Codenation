import pandas as pd

df = pd.read_csv('train.csv').fillna(0)

print(df)