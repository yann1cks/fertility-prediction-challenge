"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data,
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed,
number of folds, model, et cetera
"""

import pandas as pd
import joblib
import random
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from flaml import AutoML, tune
import joblib
from sklearn.base import clone


def stratified_group_train_test_split(
    X, y, group_col: str, n_splits: int, random_state: int | bool = None, drop_group_column=True
) -> tuple:
    """
    Wrapper around StratifiedGroupKFold for a single Train-Test-Split with grouping on the household id.
    """
    groups = X[group_col].to_list()

    if drop_group_column:
        X = X.drop(columns=[group_col])

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Get a grouped stratified Train Test Split
    train_index, test_index = next(sgkf.split(X, y, groups))

    X_train, X_test, y_train, y_test = (
        X.reset_index().loc[train_index].set_index("nomem_encr"),
        X.reset_index().loc[test_index].set_index("nomem_encr"),
        y.reset_index().loc[train_index].set_index("nomem_encr"),
        y.reset_index().loc[test_index].set_index("nomem_encr"),
    )

    return X_train, X_test, y_train.iloc[:, 0], y_test.iloc[:, 0]


def train_save_model(X, y):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """

    # This script contains a bare minimum working example
    random.seed(123)

    background = pd.read_csv(Path.cwd().parent / "data" / "other_data" / "PreFer_train_background_data.csv")

    # Sort by wave and drop duplicates afterwards. So newest wave for each individual is on top.
    nohouse_nomem_map = (
        background.sort_values("wave", ascending=False).drop_duplicates(subset=["nomem_encr"]).set_index("nomem_encr")
    )

    X = X.join(nohouse_nomem_map["nohouse_encr"])

    X_with_outcome = X[y.notnull()]
    y_with_outcome = y[y.notnull()]

    X_train, X_test, y_train, y_test = stratified_group_train_test_split(
        X_with_outcome, y_with_outcome, group_col="nohouse_encr", n_splits=4, random_state=SEED, drop_group_column=False
    )

    # Save groups to variables and drop
    train_groups = X_train["nohouse_encr"]
    test_groups = X_test["nohouse_encr"]
    X_train = X_train.drop(columns=["nohouse_encr"])
    X_test = X_test.drop(columns=["nohouse_encr"])

    # Print X Train and Test Shape
    print(f"Train Size: {len(X_train)}")
    print(f"Test Size: {len(X_test)}")

    custom_hp = {
        "lgbm": {
            "boosting_type": {"domain": "dart"},
            "data_sample_strategy": {"domain": "goss"},
            "class_weight": {"domain": tune.choice(["balanced", None])},
        },
        "xgboost": {
            "scale_pos_weight": {"domain": tune.randint(1, 30)},
        },
        "histgb": {
            "max_features": {"domain": tune.uniform(0.5, 1)},
            "class_weight": {"domain": tune.choice(["balanced", None])},
        },
    }

    automl = AutoML()

    automl.fit(
        X_train,
        y_train,
        task="classification",
        metric="f1",
        time_budget=60 * 300,
        starting_points="data",
        # max_iter=8,
        eval_method="cv",
        split_type="group",
        groups=train_groups,
        n_splits=10,
        skip_transform=False,
        estimator_list=["xgboost", "histgb", "lgbm"],
        log_file_name="flaml.log",
        ensemble=True,
        early_stop=True,
        custom_hp=custom_hp,
        seed=123,
    )

    # Retrain model and save
    final_model = clone(automl.model)
    final_model.fit(X_with_outcome, y_with_outcome)

    joblib.dump(automl, "flaml_model.joblib")
