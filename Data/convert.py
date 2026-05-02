import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")

df = df.sample(20000, random_state=42)

df = df.groupby("date")["sales"].sum().reset_index()

df.rename(columns={"sales": "demand"}, inplace=True)

df = df.sort_values("date")

df["price"] = 50 + np.random.normal(0, 5, len(df))
df["promo"] = np.random.randint(0, 2, len(df))
df["sentiment"] = np.random.uniform(-1, 1, len(df))

df["lag_1"] = df["demand"].shift(1)
df["lag_7"] = df["demand"].shift(7)
df["lag_30"] = df["demand"].shift(30)

df.dropna(inplace=True)

mean_demand = df["demand"].mean()

df["news"] = df["demand"].apply(
    lambda x: "strong increase in retail demand due to festive season and market growth"
    if x > mean_demand
    else "decline in retail demand due to economic slowdown and weak consumer activity"
)

df = df[[
    "date",
    "demand",
    "price",
    "promo",
    "sentiment",
    "lag_1",
    "lag_7",
    "lag_30",
    "news"
]]

df.to_csv("my_dataset.csv", index=False)

print("DATA READY")
