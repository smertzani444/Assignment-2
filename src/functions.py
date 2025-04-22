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
from sklearn.metrics import matthews_corrcoef, roc_auc_score, balanced_accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve, auc, confusion_matrix
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

class NestedCrossVal:
    def __init__(self):
        # Initialize models
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

        # Hyperparameter grids
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

        self.n_jobs = 1
        self.random_state = 42

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

    def model_tuning(self, model_key, X_train, y_train, inner_cv, random_state, n_jobs):
        """
        Run inner CV hyperparameter search for one model.
        Returns best fitted pipeline and params.
        """
        inner_split = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=random_state)
        best_rmse, best_pipe, best_params = float('inf'), None, None
        # Generate combos
        param_combos = self.generate_param_combinations({model_key: self.param_grid[model_key]})[model_key]
        for combo in param_combos:
            proto = self.models[model_key]
            p = proto.get_params()
            p.update(combo)
            est = proto.__class__(**p)
            pipeline = make_pipeline(StandardScaler(), est)

            aucs = cross_val_score(pipeline, X_train, y_train,
                                   scoring='roc_auc',
                                   cv=inner_split,
                                   n_jobs=self.n_jobs)
            print(f"[{model_key}] Tested {combo} -> AUC {mean_auc:.4f}")
            mean_auc = aucs.mean()
            if mean_auc > best_score:
                best_score, best_params = mean_auc, combo
                print(f"[{model_key}] New best AUC {mean_auc:.4f}, params {combo}")

        return best_pipe, best_params

    def inner_loop(self, df, target, model_key, columns_to_remove=None, inner_cv=3):
        """
        Extract train data and run model_tuning.
        """
        X_tr, y_tr = self.separate_features_target(df, target, columns_to_remove)
        return self.model_tuning(model_key, X_tr, y_tr, inner_cv)

    def outer_loop(self, df, target, model_key, outer_cv, inner_cv, columns_to_remove=None):
        """
        Outer CV: for each fold, tune hyper-params (inner), then compute test metrics.
        Returns a DataFrame of fold-wise metrics + a list of best_param dicts.
        """
        X, y = self.separate_features_target(df, target, columns_to_remove)
        outer_split = StratifiedKFold(n_splits=outer_cv,
                                      shuffle=True,
                                      random_state=self.random_state)

        records = []
        best_params_list = []

        for fold, (train_idx, test_idx) in enumerate(outer_split.split(X, y), 1):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            # 1) Inner loop: get best pipeline + params
            best_pipe, best_params = self.model_tuning(model_key, X_tr, y_tr, inner_cv)
            best_params_list.append(best_params)

            # 2) Fit on training fold
            best_pipe.fit(X_tr, y_tr)

            # 3) Predict labels & probabilities on test fold
            y_pred = best_pipe.predict(X_te)
            if hasattr(best_pipe.named_steps[best_pipe.steps[-1][0]], "predict_proba"):
                y_proba = best_pipe.predict_proba(X_te)[:,1]
            else:
                # For SVM without probability=True we can use decision_function
                y_proba = best_pipe.decision_function(X_te)

            # 4) Compute confusion matrix for specificity/NPV
            tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()

            # 5) Compute metrics
            precision, recall, _ = precision_recall_curve(y_te, y_proba)
            pr_auc = auc(recall, precision)

            rec = recall_score(y_te, y_pred)                           # sensitivity
            spec = tn / (tn + fp) if (tn + fp) else np.nan             # specificity
            npv  = tn / (tn + fn) if (tn + fn) else np.nan             # negative predictive value

            metrics = {
                'fold': fold,
                'MCC': matthews_corrcoef(y_te, y_pred),
                'AUC': roc_auc_score(y_te, y_proba),
                'PRAUC': pr_auc,
                'BA': balanced_accuracy_score(y_te, y_pred),
                'F1': f1_score(y_te, y_pred),
                'F2': fbeta_score(y_te, y_pred, beta=2),
                'Recall': rec,
                'Specificity': spec,
                'Precision': precision_score(y_te, y_pred),
                'NPV': npv
            }
            records.append(metrics)

        results_df = pd.DataFrame(records).set_index('fold')
        return results_df, best_params_list


