import optuna
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
from sklearn.metrics import average_precision_score, fbeta_score, matthews_corrcoef, roc_auc_score, balanced_accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve, auc, confusion_matrix
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

class NestedCrossVal:
    def __init__(
        self,
        R: int = 10,
        N: int = 5,
        K: int = 3,
        random_state: int = 42,
        n_jobs: int = 1
    ):
        """
        Initialize repeated nested cross-validation with default models.

        Parameters
        ----------
        R : int
            Number of repetitions of the full nested CV.
        N : int
            Number of outer folds.
        K : int
            Number of inner folds.
        random_state : int
            Seed for reproducibility.
        n_jobs : int
            Number of parallel jobs for CV.
        """
        # Initialize default models
        self.models = {
            'LogisticRegression-elasticnet': LogisticRegression(
                penalty='elasticnet', solver='saga', random_state=random_state, max_iter=10000
            ),
            'GaussianNB': GaussianNB(),
            'LDA': LinearDiscriminantAnalysis(),
            'SVC': SVC(random_state=random_state, probability=True),
            'RandomForest': RandomForestClassifier(random_state=random_state),
            'LightGBM': lgb.LGBMClassifier(random_state=random_state)
        }
        # Default hyperparameter grids
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
        self.R = R
        self.N = N
        self.K = K
        self.random_state = random_state
        self.n_jobs = n_jobs

    def generate_param_combinations(self, param_grid):
        """
        Expand param_grid into all combos per model.
        """
        # detect “winner-model” usage
        if isinstance(param_grid, list):
            param_grid = {'__winner__': param_grid}

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
        return metrics_df, best_params_list, y_te, y_pred

    def run_repeated_nested_cv(self, df, target, model_key, outer_cv, inner_cv, num_rounds, columns_to_remove=None):
        """
        Run repeated nested CV for a single specified model.

        Parameters
        ----------
        df : DataFrame
            Full dataset.
        target : str
            Name of the target column.
        model_key : str
            Key identifying which model in self.models to run.
        outer_cv : int
            Number of outer folds.
        inner_cv : int
            Number of inner folds.
        num_rounds : int
            Number of times to repeat the nested CV.
        columns_to_remove : list, optional
            Columns to drop before CV.

        Returns
        -------
        dict
            {'metrics': DataFrame of num_rounds×outer_cv rows,
             'summary': DataFrame with median and 95% CI per metric,
             'best_params': list of param dicts per outer fold per round}
        """
        all_df = []
        all_params = []
        for r in range(num_rounds):
            dfm, plist, y_te, y_pred = self.outer_loop(
                df, target, model_key,
                outer_cv, inner_cv, columns_to_remove
            )
            # bring fold index into column before concatenation
            dfm = dfm.reset_index()
            dfm['repeat'] = r + 1
            all_df.append(dfm)
            all_params.extend(plist)
        # concatenate all rounds retaining both 'fold' and 'repeat'
        full = pd.concat(all_df, ignore_index=True)
        metrics = full.drop(columns=['repeat','fold'])
        summary = pd.DataFrame({
            'median': metrics.median(),
            'ci_lower': metrics.quantile(0.025),
            'ci_upper': metrics.quantile(0.975)
        })
        return {'metrics': full, 'summary': summary, 'best_params': all_params}


class Classifier:
    def __init__(self):
        self.model = SVC(random_state=0)

        self.param_grid = [
        {'kernel': ['linear'], 'C': [0.1, 1, 10]},
        {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.01, 0.1]}
    ]

    def load_data(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"The file at {path} was not found.")
        return pd.read_csv(path)

    def preprocess_data(self, df, columns_to_drop=None):
        if columns_to_drop is None:
            columns_to_drop = []

        df=df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        num_list=df.select_dtypes(include=[np.number]).columns.tolist()
        cat_list=df.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in cat_list:
            df[col]=LabelEncoder().fit_transform(df[col])

        for col in num_list:
            df[col]=SimpleImputer(missing_values=np.nan, strategy='mean') \
                .fit_transform(df[[col]]).ravel()
    
        return df
    
    def separate_features_target(self, df, target, columns_to_remove=None):
        if columns_to_remove is None:
            columns_to_remove=[]
        columns_to_remove=set(columns_to_remove + [target])
        X=df.drop(columns=[col for col in columns_to_remove if col in df.columns])
        y=df[target]
        return X, y

    def select_features(self, X, y, threshold=0.1):
        correlations = pd.Series(r_regression(X, y), index=X.columns)
        selected_features = correlations[correlations.abs() >= threshold].index.tolist()
        print(f"The selected features of {X.shape[1]} were: {len(selected_features)}")
        return selected_features, correlations

    def generate_param_combinations(self, param_grid):
        combos = []
        # ensure we have a list of grids
        if isinstance(param_grid, list): 
            grids = param_grid 
        else:
            grids = [param_grid]

        for grid in grids:
            keys, values = zip(*grid.items())
            for vals in itertools.product(*values):
                combos.append(dict(zip(keys, vals)))

        return combos

    def model_tuning(self, model, X, y, param_grid, cv):
        best_auc    = 0.0
        best_model  = None
        best_params = None
        model_name  = model.__name__ if isinstance(model, type) else model.__class__.__name__

        combos = self.generate_param_combinations(param_grid)

        # now just loop over that list
        for combo_dict in combos:
            model_instance = model(**combo_dict)
            scores = cross_val_score(
                model_instance,
                X, y,
                scoring='roc_auc',
                cv=cv
            )
            auc = scores.mean()

            print(f"[{model_name}] Tested params: {combo_dict}")
            print(f"[{model_name}] AUC: {auc:.4f}")

            # keep this check *inside* the combo loop
            if auc > best_auc:
                best_auc    = auc
                best_model  = model_instance
                best_params = combo_dict
                print(f"[{model_name}] New best AUC: {best_auc:.4f}")
                print(f"[{model_name}] Best params so far: {best_params}")
        
        return {
            'Best AUC':    best_auc,
            'Best Model':  best_model,
            'Best Params': best_params
        }

    
    def train_tuned_model(self, model, X, y, save_path, scale=True):
        # split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # scale if requested
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
    
        # fit
        model.fit(X_train, y_train)

        # get prediction scores
        if hasattr(model, "predict_proba"):
            # binary: take probability of class “1”
            y_scores = model.predict_proba(X_test)[:, 1]
        else:
            # e.g. SVM with decision_function
            y_scores = model.decision_function(X_test)

        # compute AUC
        auc = roc_auc_score(y_test, y_scores)
        print(f"Model: {model.__class__.__name__} AUC: {auc:.4f}")

        return auc
    
    def train_final_model(self, model, X, y, save_path, scale=True):
        # optional global scaling
        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # fit on all available data
        model.fit(X, y)

        # persist
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        joblib.dump(model, save_path)
        print(f"Final model trained on all data and saved to {save_path}")

        return model

    def summarize(self, scores):
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        ci95 = t.interval(0.95, len(scores) - 1, loc=mean, scale=std / np.sqrt(len(scores)))
        return {
            'mean': mean,
            'median': np.median(scores),
            '95% CI': ci95
        }

    
    def align_evaluation_set(self, dev_df, val_df):
        dev_columns = dev_df.columns
        val_aligned = val_df.copy()
        val_aligned = val_aligned.reindex(columns=dev_columns, fill_value=0)
        print("Evaluation dataset was aligned to development feature set.")
        return val_aligned
    

    def evaluate_model(self, model, X, y, runs=30, test_size=0.2, scale=True, save_path=None):
        metrics = {
            'auc':   [],
            'prauc': [],
            'mcc':   []
        }

        best_auc = 0.0
        best_model = None

        for i in range(runs):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            if scale:
                scaler=StandardScaler()
                X_train=scaler.fit_transform(X_train)
                model.fit(X_train, y_train)
                X_test=scaler.fit_transform(X_test)
                y_pred = model.predict(X_test)
             
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(X_test)[:, 1]
            else:
                y_scores = model.decision_function(X_test)

            auc_val   = roc_auc_score(y_test, y_scores)
            prauc_val = average_precision_score(y_test, y_scores)
            mcc_val   = matthews_corrcoef(y_test, y_pred)
            
            metrics['auc'].append(auc_val)
            metrics['prauc'].append(prauc_val)
            metrics['mcc'].append(mcc_val)
            
            print(f"[Run {i+1}] AUC: {auc_val:.4f}, PR AUC: {prauc_val:.4f}, MCC: {mcc_val:.4f}")
            
            # Track the best model
            if auc_val > best_auc:
                best_auc   = auc_val
                best_model = np.copy.deepcopy(model)
                print(f"  ↳ New best model (AUC={best_auc:.4f})")
        
        results = {k: self.summarize(v) for k, v in metrics.items()}

        if save_path is not None and best_model is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(best_model, save_path)
            print(f"Best model (highest AUC: {best_auc:.4f}) saved to {save_path}")

        for metric, values in metrics.items():
            plt.figure(figsize=(8, 6))
            sns.boxplot(y=values)
            plt.title(f"{metric.upper()} Distribution")
            plt.ylabel(metric.upper())
            plt.show()

        return results



