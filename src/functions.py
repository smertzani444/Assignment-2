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
from sklearn.metrics import fbeta_score, matthews_corrcoef, roc_auc_score, balanced_accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve, auc, confusion_matrix
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

class NestedCrossVal:
    def __init__(self,
                 models: dict,
                 param_grid: dict,
                 R: int = 10,
                 N: int = 5,
                 K: int = 3,
                 random_state: int = 42,
                 n_jobs: int = 1):
        """
        Initialize the repeated nested cross-validation runner.

        Parameters
        ----------
        models : dict
            Mapping of model names to unfitted estimator instances.
        param_grid : dict
            Mapping of model names to their hyperparameter grids.
        R : int
            Number of repetitions of the nested CV process (outer+inner).
        N : int
            Number of folds in the outer loop.
        K : int
            Number of folds in the inner loop.
        random_state : int
            Seed for reproducibility in splitting.
        n_jobs : int
            Number of parallel jobs to run.
        """
        self.models = models
        self.param_grid = param_grid
        self.R = R
        self.N = N
        self.K = K
        self.random_state = random_state
        self.n_jobs = n_jobs

    def generate_param_combinations(self, param_grid):
        """
        Expand param_grid into all combos per model.
        """
        model_combinations = {}
        for model, params in param_grid.items():
            if isinstance(params, list):
                combos = []
                for pset in params:
                    keys, vals = zip(*pset.items())
                    for v in itertools.product(*vals):
                        combos.append(dict(zip(keys, v)))
                model_combinations[model] = combos
            else:
                keys, vals = zip(*params.items())
                model_combinations[model] = [dict(zip(keys, v)) for v in itertools.product(*vals)]
        return model_combinations

    def separate_features_target(self, df, target, columns_to_remove=None):
        """
        Splits DataFrame into feature matrix X and target y.
        """
        cols_to_drop = set(columns_to_remove or []) | {target}
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        y = df[target]
        return X.values, y.values

    def model_tuning(self, model_key, X_train, y_train, inner_cv):
        """
        Run inner CV hyperparameter search for one model.
        Uses self.random_state and self.n_jobs internally.
        """
        inner_split = StratifiedKFold(
            n_splits=inner_cv,
            shuffle=True,
            random_state=self.random_state
        )

        if model_key == 'LightGBM':
            # compute imbalance ratio on this fold's training data
            neg, pos = np.bincount(y_train)
            ratio = neg / pos
            # make a shallow copy of the param grid so we don't pollute self.param_grid permanently
            local_grid = dict(self.param_grid['LightGBM'])
            local_grid.update({
                'scale_pos_weight': [ratio, 1.0],
                'boosting_type':      ['gbdt', 'dart'],
                'reg_alpha':          [0, 0.1, 1],
                'reg_lambda':         [0, 0.1, 1],
                # you can also experiment with more n_estimators here
            })
        else:
            local_grid = self.param_grid[model_key]

        best_score = -np.inf
        best_pipe = None
        best_params = None

        param_combos = self.generate_param_combinations(
            {model_key: local_grid}
        )[model_key]

        for combo in param_combos:
            proto = self.models[model_key]
            p = proto.get_params()
            p.update(combo)
            est = proto.__class__(**p)
            pipeline = make_pipeline(StandardScaler(), est)

            aucs = cross_val_score(
                pipeline,
                X_train,
                y_train,
                scoring='roc_auc',
                cv=inner_split,
                n_jobs=self.n_jobs
            )
            mean_auc = aucs.mean()
            print(f"[{model_key}] Tested {combo} -> AUC {mean_auc:.4f}")

            if mean_auc > best_score:
                best_score, best_params = mean_auc, combo
                best_pipe = pipeline
                print(f"[{model_key}] New best AUC {mean_auc:.4f}, params {combo}")

        # Return the (unfitted) pipeline and the best params dict:
        return best_pipe, best_params

    def inner_loop(self, df, target, model_key, columns_to_remove=None, inner_cv=3):
        """
        Extract train data and run model_tuning.
        """
        X_tr, y_tr = self.separate_features_target(df, target, columns_to_remove)
        return self.model_tuning(model_key, X_tr, y_tr, inner_cv)

    def outer_loop(self, df, target, model_key, outer_cv, inner_cv, columns_to_remove=None):
   
        # Determine which cols to drop
        columns_to_remove = set(columns_to_remove or []) | {target}
        feature_cols = [c for c in df.columns if c not in columns_to_remove]

        X_df = df[feature_cols]
        y_sr = df[target]

        outer_split = StratifiedKFold(
            n_splits=outer_cv,
            shuffle=True,
            random_state=self.random_state
        )

        records = []
        best_params_list = []

        for fold, (train_idx, test_idx) in enumerate(outer_split.split(X_df, y_sr), start=1):
            # Build fold DataFrames/Series
            X_tr = X_df.iloc[train_idx]
            y_tr = y_sr.iloc[train_idx]
            X_te = X_df.iloc[test_idx]
            y_te = y_sr.iloc[test_idx]

            # Inner tuning: get the best pipeline and its params
            best_pipe, best_params = self.model_tuning(model_key, X_tr, y_tr, inner_cv)
            best_params_list.append(best_params)

            # Fit on the training fold
            best_pipe.fit(X_tr, y_tr)

            # Predictions
            y_pred = best_pipe.predict(X_te)
            if hasattr(best_pipe, "predict_proba"):
                y_proba = best_pipe.predict_proba(X_te)[:, 1]
            else:
                y_proba = best_pipe.decision_function(X_te)

            # Confusion-matrix stats
            tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
            recall = recall_score(y_te, y_pred)
            specificity = tn / (tn + fp) if (tn + fp) else np.nan
            npv = tn / (tn + fn) if (tn + fn) else np.nan

            # Precision–Recall AUC
            prec_vals, rec_vals, _ = precision_recall_curve(y_te, y_proba)
            pr_auc = auc(rec_vals, prec_vals)

            # Collect all metrics
            records.append({
                'fold': fold,
                'MCC': matthews_corrcoef(y_te, y_pred),
                'AUC': roc_auc_score(y_te, y_proba),
                'PRAUC': pr_auc,
                'BA': balanced_accuracy_score(y_te, y_pred),
                'F1': f1_score(y_te, y_pred),
                'F2': fbeta_score(y_te, y_pred, beta=2),
                'Recall': recall,
                'Specificity': specificity,
                'Precision': precision_score(y_te, y_pred),
                'NPV': npv
            })

        metrics_df = pd.DataFrame(records).set_index('fold')
        return metrics_df, best_params_list

    def run_repeated_nested_cv(self, df, target, columns_to_remove=None):
        """
        Execute R repetitions of nested CV across all models.

        Returns
        -------
        results : dict
            For each model:
              - 'metrics': DataFrame of R×N outer-fold results
              - 'summary': DataFrame with median, ci_lower, ci_upper per metric
              - 'best_params': list of best_params per outer fold
        """
        results = {}
        for model_key in self.models:
            all_metrics = []
            all_params = []
            for r in range(self.R):
                df_metrics, params_list = self.outer_loop(
                    df, target, model_key,
                    columns_to_remove=columns_to_remove
                )
                df_metrics['repeat'] = r
                all_metrics.append(df_metrics)
                all_params.extend(params_list)

            full_df = pd.concat(all_metrics, ignore_index=True)
            # Calculate medians and 95% CI
            metrics_only = full_df.drop(columns=['repeat', 'fold'])
            median = metrics_only.median()
            ci_lower = metrics_only.quantile(0.025)
            ci_upper = metrics_only.quantile(0.975)
            summary = pd.DataFrame({
                'median': median,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            })

            results[model_key] = {
                'metrics': full_df,
                'summary': summary,
                'best_params': all_params
            }
        return results





