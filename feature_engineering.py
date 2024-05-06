"""Code detailing our final feature engineering.

Thank you to https://www.kaggle.com/competitions/playground-series-s4e5/discussion/499274!
"""

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
features = list(train_data)
features.remove("FloodProbability")


for df in [train_data, test_data]:
    # From  https://www.kaggle.com/competitions/playground-series-s4e5/discussion/499274
    df["fsum"] = df[features].sum(axis=1)
    df["f_std"] = df[features].std(axis=1)
    df["special1"] = df["fsum"].isin(np.arange(72, 76))

    # Ours
    df["trim_mean"] = trim_mean(
        df[features],
        proportiontocut=0.075,
        axis=1,
    )
    df["special1"] = df["special1"].astype("category")
