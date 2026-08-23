"""The candidate models and the hyperparameter space searched for each.

Parameter keys are prefixed 'model__' because every estimator is the final
step of a Pipeline whose first step is the preprocessor.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from backend.core.config import settings

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False

RS = settings.RANDOM_STATE


def classification_models() -> dict[str, dict]:
    zoo = {
        "LogisticRegression": {
            "estimator": LogisticRegression(max_iter=2000, random_state=RS),
            "params": {
                "model__C": np.logspace(-3, 2, 20),
                "model__penalty": ["l2"],
                "model__class_weight": [None, "balanced"],
            },
            "cost": 1,
        },
        "SVC": {
            # probability=True is needed for ROC-AUC, at the cost of an
            # internal cross-validation on every fit.
            "estimator": SVC(probability=True, random_state=RS),
            "params": {
                "model__C": np.logspace(-2, 2, 12),
                "model__kernel": ["rbf", "linear", "poly"],
                "model__gamma": ["scale", "auto", 0.01, 0.1],
                "model__class_weight": [None, "balanced"],
            },
            "cost": 5,
        },
        "DecisionTree": {
            "estimator": DecisionTreeClassifier(random_state=RS),
            "params": {
                "model__max_depth": [None, 4, 6, 10, 16],
                "model__min_samples_split": [2, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__criterion": ["gini", "entropy"],
            },
            "cost": 1,
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=RS, n_jobs=-1),
            "params": {
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [None, 6, 12, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__max_features": ["sqrt", "log2", None],
                "model__class_weight": [None, "balanced"],
            },
            "cost": 3,
        },
        "GradientBoosting": {
            "estimator": GradientBoostingClassifier(random_state=RS),
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__max_depth": [2, 3, 5],
                "model__subsample": [0.8, 1.0],
            },
            "cost": 4,
        },
        "KNeighbors": {
            "estimator": KNeighborsClassifier(),
            "params": {
                "model__n_neighbors": [3, 5, 7, 11, 15, 21],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            "cost": 2,
        },
        "GaussianNB": {
            "estimator": GaussianNB(),
            "params": {"model__var_smoothing": np.logspace(-11, -6, 12)},
            "cost": 1,
        },
    }
    if HAS_XGBOOST:
        zoo["XGBoost"] = {
            "estimator": XGBClassifier(
                random_state=RS, eval_metric="logloss",
                tree_method="hist", verbosity=0,
            ),
            "params": {
                "model__n_estimators": [100, 200, 400],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__max_depth": [3, 5, 7, 9],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
            },
            "cost": 3,
        }
    return zoo


def regression_models() -> dict[str, dict]:
    zoo = {
        "LinearRegression": {
            "estimator": LinearRegression(),
            "params": {"model__fit_intercept": [True, False]},
            "cost": 1,
        },
        "Ridge": {
            "estimator": Ridge(random_state=RS),
            "params": {
                "model__alpha": np.logspace(-3, 3, 20),
                "model__fit_intercept": [True, False],
            },
            "cost": 1,
        },
        "Lasso": {
            "estimator": Lasso(random_state=RS, max_iter=5000),
            "params": {"model__alpha": np.logspace(-4, 1, 20)},
            "cost": 1,
        },
        "ElasticNet": {
            "estimator": ElasticNet(random_state=RS, max_iter=5000),
            "params": {
                "model__alpha": np.logspace(-4, 1, 15),
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
            "cost": 1,
        },
        "SVR": {
            "estimator": SVR(),
            "params": {
                "model__C": np.logspace(-2, 2, 12),
                "model__kernel": ["rbf", "linear", "poly"],
                "model__gamma": ["scale", "auto", 0.01, 0.1],
                "model__epsilon": [0.01, 0.1, 0.2, 0.5],
            },
            "cost": 5,
            # RBF/poly SVR assumes a target near unit scale; without this a
            # price column in the hundreds of thousands scores near zero.
            "scale_target": True,
        },
        "DecisionTree": {
            "estimator": DecisionTreeRegressor(random_state=RS),
            "params": {
                "model__max_depth": [None, 4, 6, 10, 16],
                "model__min_samples_split": [2, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 4, 8],
            },
            "cost": 1,
        },
        "RandomForest": {
            "estimator": RandomForestRegressor(random_state=RS, n_jobs=-1),
            "params": {
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [None, 6, 12, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__max_features": ["sqrt", "log2", None],
            },
            "cost": 3,
        },
        "GradientBoosting": {
            "estimator": GradientBoostingRegressor(random_state=RS),
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__max_depth": [2, 3, 5],
                "model__subsample": [0.8, 1.0],
            },
            "cost": 4,
        },
        "KNeighbors": {
            "estimator": KNeighborsRegressor(),
            "params": {
                "model__n_neighbors": [3, 5, 7, 11, 15, 21],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            "cost": 2,
            "scale_target": True,
        },
    }
    if HAS_XGBOOST:
        zoo["XGBoost"] = {
            "estimator": XGBRegressor(
                random_state=RS, tree_method="hist", verbosity=0,
            ),
            "params": {
                "model__n_estimators": [100, 200, 400],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__max_depth": [3, 5, 7, 9],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
            },
            "cost": 3,
        }
    return zoo


def get_zoo(task: str) -> dict[str, dict]:
    from backend.training.task_detect import CLASSIFICATION
    return classification_models() if task == CLASSIFICATION else regression_models()


def available_models(task: str) -> list[str]:
    return list(get_zoo(task).keys())


def select_models(task: str, requested: list[str] | None, n_rows: int) -> dict[str, dict]:
    """Honour an explicit list, otherwise drop the models whose cost is
    unreasonable at this dataset size."""
    zoo = get_zoo(task)
    if requested:
        unknown = [m for m in requested if m not in zoo]
        if unknown:
            from backend.core.exceptions import ValidationError
            raise ValidationError(
                f"Unknown model(s): {unknown}",
                detail=f"Available for {task}: {list(zoo)}",
            )
        return {name: zoo[name] for name in requested}

    if n_rows > 50_000:
        return {n: c for n, c in zoo.items() if c["cost"] <= 3}
    if n_rows > 10_000:
        return {n: c for n, c in zoo.items() if c["cost"] <= 4}
    return zoo
