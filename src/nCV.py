import pandas as pd
import numpy as np
import joblib
import sys
import os
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import r_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline 
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.base import clone
from typing import Any, Callable, Dict, List, Tuple, Union
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

class NestedCrossVal:
    def __init__(self):
        self.models = {
            'LogisticRegression-elasticnet': LogisticRegression(
            penalty='elasticnet', solver='saga', random_state=0, max_iter=10000
            ),
            'GaussianNB': GaussianNB(),
            'LDA': LinearDiscriminantAnalysis(),
            'SVC': SVC(random_state=0),
            'RandomForest': RandomForestClassifier(random_state=0),
            'LightGBM': LGBMClassifier(random_state=0)
        }

        self.param_grid = {
            'LogisticRegression-elasticnet': {
                'C': [0.01, 0.1, 1, 10],
                'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
            },
            'GaussianNB': {
                'var_smoothing': np.logspace(-9, -1, 9)
            },
            'LDA': [
                {'solver': ['svd']},
                {'solver': ['lsqr', 'eigen'], 'shrinkage': [None, 'auto']}
            ],
            'SVC': [
                {'kernel': ['linear'], 'C': [0.1, 1, 10]},
                {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.01, 0.1]}
            ],
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'LightGBM': {
                'n_estimators': [100, 200],
                'num_leaves': [31, 50, 100],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [-1, 10, 20]
            }
        }

    def generate_param_combinations(self, param_grid):
        model_combinations = {}
        for model, params in param_grid.items():
        
            if isinstance(params, list):
                param_combinations = []
                for param_set in params:
                    param_combinations.extend(
                        [dict(zip(param_set.keys(), values)) for values in itertools.product(*param_set.values())]
                    )
                model_combinations[model] = param_combinations
            else:
                model_combinations[model] = [
                    dict(zip(params.keys(), values)) for values in itertools.product(*params.values())
                ]
        return model_combinations
    
    def separate_features_target(self, df, target, columns_to_remove=None):
        if columns_to_remove is None:
            columns_to_remove=[]
        columns_to_remove=set(columns_to_remove + [target])
        X=df.drop(columns=[col for col in columns_to_remove if col in df.columns])
        y=df[target]
        return X, y
    
    def model_tuning(self, model_key, X_train, y_train, inner_cv=3, random_state=42, n_jobs=1):
        splitter = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=random_state)
        best_rmse = float('inf')
        best_model = None
        best_params = None

        param_combinations = self.generate_param_combinations({model_key: self.param_grid[model_key]})[model_key]


        for combination in param_combinations:
            proto = self.models[model_key]
            proto_params = proto.get_params()
            proto_params.update(combination)
            estimator = proto.__class__(**proto_params)

            pipeline = make_pipeline(StandardScaler(), estimator)
            scores = cross_val_score(
                pipeline, X_train, y_train,
                scoring='neg_root_mean_squared_error', cv=splitter, n_jobs=n_jobs
            )
            rmse = np.sqrt(-scores.mean())
            print(f"[{model_key}] Tested {combination} -> RMSE {rmse:.4f}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = pipeline
                best_params = combination
                print(f"[{model_key}] New best RMSE {best_rmse:.4f}, params {best_params}")

        return best_model, best_params
    
    def inner_loop(self, df, target, model_key, columns_to_remove=None, inner_cv=3):
        X, y = self.separate_features_target(df, target, columns_to_remove)
        return self.model_tuning(model_key, X.values, y.values, inner_cv)
    
    def outer_loop(self, df, target, model_key, outer_cv=5, random_state=42, columns_to_remove=None, inner_cv=3):
        # Separate full features and target once
        X_full, y_full = self.separate_features_target(df, target, columns_to_remove)
        splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
        scores = []
        params_per_fold = []

        for train_idx, test_idx in splitter.split(X_full.values, y_full.values):
            # Partition DataFrame
            df_tr = df.iloc[train_idx].reset_index(drop=True)
            df_te = df.iloc[test_idx].reset_index(drop=True)

            # Inner tuning on training partition
            best_pipe, best_params = self.inner_loop(df_tr, target, model_key, columns_to_remove, inner_cv)
            params_per_fold.append(best_params)

            # Extract train arrays for fitting
            X_tr, y_tr = self.separate_features_target(df_tr, target, columns_to_remove)
            X_tr_arr, y_tr_arr = X_tr.values, y_tr.values

            # Extract test arrays for evaluation
            X_te, y_te = self.separate_features_target(df_te, target, columns_to_remove)
            X_te_arr, y_te_arr = X_te.values, y_te.values

            # Fit pipeline on training data arrays and score on test arrays
            best_pipe.fit(X_tr_arr, y_tr_arr)
            score = best_pipe.score(X_te_arr, y_te_arr)
            scores.append(score)

        return scores, params_per_fold
