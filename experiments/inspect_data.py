import numpy as np
import pandas as pd
import json
from pathlib import Path

datadir = Path("dataset_2022_silverstone_ham")

X = np.load(datadir/"X.npy")
t = np.load(datadir/"t.npy")
lap_idx = np.load(datadir/"lap_indices.npy")

with open(datadir/"metadata.json") as f:
    meta = json.load(f)

state_cols = meta["state_cols"]

scaler = json.load(open(datadir / "scaler.json"))

X_mean = np.array(scaler["X_mean"])
X_scale = np.array(scaler["X_scale"])

X_phys = X * X_scale + X_mean
df_phys = pd.DataFrame(X_phys, columns=state_cols)
df_phys["t"] = t
df_phys["lap"] = lap_idx

df_phys.to_csv(datadir / "dataset_physical_units.csv", index=False)
print(df_phys.head())
print(df_phys.info())
print(X_phys.shape)
print(t.shape)
print(lap_idx.shape)