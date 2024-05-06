"""Code detailing how we use AutoGluon as a framework (and not a tool) for mode selection."""

from autogluon.tabular import TabularPredictor

# Hyperparameters taken from TabRepo - https://github.com/autogluon/tabrepo
#   - We selected only models that were fast to fit and seemed to perform well on the dataset.
#   - Due to the size of the dataset we filtered to the first two configs for each model type.
CUSTOM_HYPERPARAMETERS = {
    "GBM": [
        {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
        "GBMLarge",
    ],
    "CAT": [
        {},
        {
            "depth": 6,
            "grow_policy": "SymmetricTree",
            "l2_leaf_reg": 2.1542798306067823,
            "learning_rate": 0.06864209415792857,
            "max_ctr_complexity": 4,
            "one_hot_max_size": 10,
            "ag_args": {"name_suffix": "_r177", "priority": -1},
        },
    ],
    "XGB": [
        {},
        {
            "colsample_bytree": 0.6917311125174739,
            "enable_categorical": False,
            "learning_rate": 0.018063876087523967,
            "max_depth": 10,
            "min_child_weight": 0.6028633586934382,
            "ag_args": {"name_suffix": "_r33", "priority": -9},
        },
    ],
    "XT": [
        {"criterion": "squared_error", "ag_args": {"name_suffix": "MSE", "problem_types": ["regression", "quantile"]}},
        {
            "max_features": 1.0,
            "max_leaf_nodes": 18729,
            "min_samples_leaf": 5,
            "ag_args": {"name_suffix": "_r137", "priority": -16},
        },
    ],
    "LR": [
        {
            "C": 978.6204803985407,
            "penalty": "L2",
            "proc.impute_strategy": "mean",
            "proc.skew_threshold": None,
            "ag_args": {"name_suffix": "_r9", "priority": 999},
        },
        {
            "C": 958.9533736976579,
            "penalty": "L1",
            "proc.impute_strategy": "mean",
            "proc.skew_threshold": 0.99,
            "ag_args": {"name_suffix": "_r22", "priority": 999},
        },
    ],
}


# -- Default Fit
predictor = TabularPredictor(
    label="FloodProbability",
    eval_metric="r2",
    problem_type="regression",
    verbosity=2,
)

predictor.fit(
    hyperparameters=CUSTOM_HYPERPARAMETERS,
    # Resources
    time_limit=4 * 60 * 60,
    num_cpus=32,
    num_gpus=0,
    # Validation Protocol
    num_bag_folds=8,
    num_bag_sets=5,
    num_stack_levels=4,
    # Dynamic Stacking
    dynamic_stacking=False,
    # Other
    presets="best_quality",
)
predictor.fit_summary(verbosity=1)

# -- Fit Extra Code used after Loading the Predictor from Disk to Fit Extra Models to the Existing Predictor
# Useful when you want to add more models to an existing predictor under a time budget.
predictor.fit_extra(
    hyperparameters=CUSTOM_HYPERPARAMETERS,
    time_limit=60 * 60,
    base_model_names=[
        # Add here names of models to use as base models for stacking.
        # We always selected the last layer's models as base models (names taken from the leaderboard).
    ],
    num_cpus=32,
    num_gpus=0,
)


# -- Finally we added the following hacks to AutoGluon by editing the code base in our local library:
#   - Set the number of iterations (DEFAULT_NUM_BOOST_ROUND) for each GBDT algorithm from 10k to 50k due to the size of
#     the dataset.
#   - Set the number of iterations from 25 to 100 in greedy ensemble selection (the weighted ensemble).
#   - Added `r2="l2",` to the objective function map of LightGBM. This was a bug in AutoGluon and this fix will be
#     merged.
