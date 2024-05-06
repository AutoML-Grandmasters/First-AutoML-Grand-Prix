"""Code detailing how one can merge two AutoGluon submissions to (hopefully) get a better score."""

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import r2_score

# Get Previous AutoGluon Runs
old_best_predictor = TabularPredictor.load("/path/to/first/AutoGluon/run")
curr_best_predictor = TabularPredictor.load("/path/to/second/AutoGluon/run")

old_oof = old_best_predictor.predict_proba_oof()
new_oof = curr_best_predictor.predict_proba_oof()
_, label = curr_best_predictor.load_data_internal(return_X=False)


# -- Find Best Merge Weights
print("Old", r2_score(label, old_oof))
new_score = r2_score(label, new_oof)
print("New", new_score)

res_list = []
for i in np.arange(0, 1, 0.01):
    score = r2_score(label, (old_oof * i) + (new_oof * (1 - i)))
    if score > new_score:
        res_list.append([i, score])

df = pd.DataFrame(res_list, columns=["i", "score"])
print(df.sort_values(by="score", ascending=False).head(10))


# -- Merge Predictions
i = df.sort_values(by="score", ascending=False).iloc[0]["i"]
old_preds = pd.read_csv("/path/to/first/AutoGluon/run/predictions")
new_preds = pd.read_csv("/path/to/second/AutoGluon/run/predictions")
new_preds["FloodProbability"] = (old_preds["FloodProbability"] * i) + (new_preds["FloodProbability"] * (1 - i))
new_preds.to_csv("./final_merge_test.csv", index=False)
