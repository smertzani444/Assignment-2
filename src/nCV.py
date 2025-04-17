import pandas as pd
import numpy as np
import joblib
import sys
import os
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t
import pprint
import copy
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import r_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

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
            'LightGBM': lgb.LGBMClassifier(random_state=0)
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

    def generate_param_combintions(param_grid):
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
    
    def separate_features_target(df, target, columns_to_remove=None):
        if columns_to_remove is None:
            columns_to_remove=[]
        columns_to_remove=set(columns_to_remove + [target])
        X=df.drop(columns=[col for col in columns_to_remove if col in df.columns])
        y=df[target]
        return X, y