import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    matthews_corrcoef, recall_score, fbeta_score,
    balanced_accuracy_score, f1_score, precision_score,
    confusion_matrix
)
import warnings

warnings.filterwarnings("ignore")  # suppress sklearn and other warnings

class LightGBMNestedCV:
    """
    Nested cross-validation for a single LightGBM model.
    """
    DEFAULT_PARAM_GRID = {
        'n_estimators': [100, 200],
        'num_leaves': [31, 50, 100],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [-1, 10, 20]
    }

    def __init__(
        self,
        param_grid: dict = None,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Parameters
        ----------
        param_grid : dict, optional
            Hyperparameter grid for LightGBM. If None, uses DEFAULT_PARAM_GRID.
        random_state : int
            Random seed for reproducibility.
        n_jobs : int
            Number of parallel jobs for CV.
        """
        # Use provided grid or default
        if param_grid is None:
            self.param_grid = dict(self.DEFAULT_PARAM_GRID)
        else:
            self.param_grid = param_grid
        self.random_state = random_state
        self.n_jobs = n_jobs
        """
        Parameters
        ----------
        param_grid : dict, optional
            Hyperparameter grid for LightGBM. If None, uses DEFAULT_PARAM_GRID.
        random_state : int
            Random seed for reproducibility.
        n_jobs : int
            Number of parallel jobs for CV (currently unused).
        """
        # Use provided grid or default
        self.param_grid = param_grid if param_grid is not None else dict(self.DEFAULT_PARAM_GRID)
        self.random_state = random_state
        self.n_jobs = n_jobs

    def model_tuning(self, X_train, y_train, inner_cv: int):
        """
        Inner CV grid search over LightGBM hyperparameters.
        Returns best unfitted LGBMClassifier and its params.
        """
        inner_split = StratifiedKFold(
            n_splits=inner_cv,
            shuffle=True,
            random_state=self.random_state
        )

        # Compute imbalance ratio
        neg, pos = np.bincount(y_train)
        ratio = neg / pos if pos else 1.0

        # Build local grid including scale_pos_weight
        local_grid = dict(self.param_grid)
        local_grid['scale_pos_weight'] = [ratio, 1.0]

        # Generate all combinations
        from itertools import product
        keys, values = zip(*local_grid.items())
        combos = [dict(zip(keys, v)) for v in product(*values)]

        best_score = -np.inf
        best_params = None
        best_clf = None

        for combo in combos:
            clf = lgb.LGBMClassifier(
                **combo,
                objective='binary',
                verbosity=-1,
                random_state=self.random_state
            )
            # Cross-validated AUC without early stopping
            scores = []
            for tr_idx, val_idx in inner_split.split(X_train, y_train):
                # Fit on training fold
                clf.fit(
                    X_train.iloc[tr_idx], y_train.iloc[tr_idx]
                )
                # Predict probabilities on validation fold
                y_pred = clf.predict_proba(X_train.iloc[val_idx])[:, 1]
                scores.append(roc_auc_score(y_train.iloc[val_idx], y_pred))
            mean_auc = np.mean(scores)

            if mean_auc > best_score:
                best_score = mean_auc
                best_params = combo
                best_clf = lgb.LGBMClassifier(
                    **combo,
                    objective='binary',
                    verbosity=-1,
                    random_state=self.random_state
                )
        return best_clf, best_params

    def outer_loop(
        self,
        df: pd.DataFrame,
        target: str,
        outer_cv: int,
        inner_cv: int,
        columns_to_remove: list = None
    ):
        """
        Perform nested CV: outer loop for evaluation and inner for tuning.
        Returns metrics DataFrame and list of best params per fold.
        """
        columns_to_remove = set(columns_to_remove or []) | {target}
        feature_cols = [c for c in df.columns if c not in columns_to_remove]

        X = df[feature_cols]
        y = df[target]

        outer_split = StratifiedKFold(
            n_splits=outer_cv,
            shuffle=True,
            random_state=self.random_state
        )

        records = []
        best_params_list = []

        for fold, (train_idx, test_idx) in enumerate(
            outer_split.split(X, y), start=1
        ):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

            # Inner tuning
            best_clf, best_params = self.model_tuning(X_tr, y_tr, inner_cv)
            best_params_list.append(best_params)

            # Fit on full train fold
            best_clf.fit(X_tr, y_tr)

            # Predictions
            y_pred = best_clf.predict(X_te)
            y_proba = best_clf.predict_proba(X_te)[:, 1]

            tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
            prec, rec, _ = precision_recall_curve(y_te, y_proba)
            pr_auc = auc(rec, prec)

            records.append({
                'fold': fold,
                'MCC': matthews_corrcoef(y_te, y_pred),
                'AUC': roc_auc_score(y_te, y_proba),
                'PRAUC': pr_auc,
                'BA': balanced_accuracy_score(y_te, y_pred),
                'F1': f1_score(y_te, y_pred),
                'F2': fbeta_score(y_te, y_pred, beta=2),
                'Recall': recall_score(y_te, y_pred),
                'Specificity': tn / (tn + fp) if (tn + fp) else np.nan,
                'Precision': precision_score(y_te, y_pred),
                'NPV': tn / (tn + fn) if (tn + fn) else np.nan
            })

        metrics_df = pd.DataFrame(records).set_index('fold')
        return metrics_df, best_params_list

    def run_repeated_nested_cv(
        self,
        df: pd.DataFrame,
        target: str,
        outer_cv: int,
        inner_cv: int,
        num_rounds: int = 1,
        columns_to_remove: list = None
    ):
        """
        Repeat nested CV multiple times for stability.
        Returns full metrics, summary, and all best params.
        """
        all_dfs = []
        all_params = []

        for r in range(1, num_rounds + 1):
            dfm, plist = self.outer_loop(
                df, target, outer_cv, inner_cv, columns_to_remove
            )
            dfm = dfm.reset_index()
            dfm['repeat'] = r
            all_dfs.append(dfm)
            all_params.extend(plist)

        full = pd.concat(all_dfs, ignore_index=True)
        metrics = full.drop(columns=['repeat', 'fold'])
        summary = pd.DataFrame({
            'median': metrics.median(),
            'ci_lower': metrics.quantile(0.025),
            'ci_upper': metrics.quantile(0.975)
        })

        return {'metrics': full, 'summary': summary, 'best_params': all_params}
