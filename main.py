import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

file_name = "heart_final_train.csv"
df = pd.read_csv(file_name, engine="python", encoding="utf-8-sig")

df.dropna(axis=0, inplace=True)

categorical_features = ["cp", "restecg", "slp", "thall"]
df = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features)

df_numerical = df.select_dtypes(include=["number"])
plt.figure(figsize=(10, 8))
sns.heatmap(df_numerical.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.show()

df_numerical = df.select_dtypes(include=["number"])
df_numerical.corr()["output"].sort_values(ascending=False)