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

# **Note that the methods designed for the previous assignment that were employed
# in this assignment may have been slightly modifiedin order to fit in this assignment as well or slightly improved 
# **Note that the methods designed for the previous assignment that were employed
# in this assignment take as input the argument **model**, whereas the methods created for this assignment 
# take as input the argument **model_key**, which is ultimately the same

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
            Number of rounds for the nCV.
        N : int
            Number of outer fold loops.
        K : int
            Number of inner fold loops.
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
        
        # Default hyperparameter spaces for each classifier that will be used for tuning
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
        
        # prepare the output dictionary 
        # keys are model names, values are list of hyperparameters combination dicts
        model_combinations = {}

        # iterate over each model name and its associated hyperparameters combinations 
        # iterate over each (model, params) pair in the param_grid
        for model, params in param_grid.items():
            # if params is a list of multiple grids,
            # handle each grid separately
            if isinstance(params, list):
                combos = []
                # iterate through each parameter grid dictionary in the list
                # pset = set of parameters 
                for pset in params:
                    # concept of itertools.product -> we used it on the unpacked lists of values to 
                    # generate every combination and then zip back with keys to yield concrete parameter-setting dict
                    # unzip the dict into parallel sequences of keys and lists-of-values
                    keys, vals = zip(*pset.items())
                    for v in itertools.product(*vals):
                        # zip back into a dict mapping each key to one chosen value
                        combos.append(dict(zip(keys, v)))
                model_combinations[model] = combos

            # if params is a single dict, expand directly
            else:
                # separate parameter names and their candidate values
                keys, vals = zip(*params.items())
                # build one config dict per combination of values
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
        # we used StratifiedKFold because we took into account that the dataset was slightly imbalanced and we want the folds
        # we crate to reflect the whole dataset's class imbalance
        # shuffle=True, shuffle before splitting
        inner_split = StratifiedKFold(
            n_splits=inner_cv,
            shuffle=True,
            random_state=self.random_state
        )
        
        local_grid = self.param_grid[model_key]
        
        # initialize best score, best pipe and best parameters 
        best_score = -np.inf
        best_pipe = None
        best_params = None
        
        # employ generate_param_combinations method 
        param_combos = self.generate_param_combinations(
            {model_key: local_grid}
        )[model_key]

        # iterate over every hyperparameter combination
        for combo in param_combos:
            proto = self.models[model_key]
            p = proto.get_params()
            p.update(combo)
            est = proto.__class__(**p)
            # build a small pipeline: standardize → estimator
            pipeline = make_pipeline(StandardScaler(), est)
            
            # evaluate via cross‐validation using ROC AUC
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
            
            # update the “current best” if this combo results in greater mean auc score than the previous best score
            if mean_auc > best_score:
                best_score, best_params = mean_auc, combo
                best_pipe = pipeline
                print(f"[{model_key}] New best AUC {mean_auc:.4f}, params {combo}")

        # return the best (unfitted) pipeline and the best params dict:
        return best_pipe, best_params

    def outer_loop(self, df, target, model_key, outer_cv, inner_cv, columns_to_remove=None):
        """
        Run outer CV performasnce evaluation for one model.
        Uses self.random_state and self.n_jobs internally.
        """

        # determine which cols to drop
        columns_to_remove = set(columns_to_remove or []) | {target}
        feature_cols = [c for c in df.columns if c not in columns_to_remove]

        X = df[feature_cols]
        y = df[target]

        # we used StratifiedKFold because we took into account that the dataset was slightly imbalanced and we want the folds
        # we crate to reflect the whole dataset's class imbalance
        # shuffle=True, shuffle before splitting
        outer_split = StratifiedKFold(
            n_splits=outer_cv,
            shuffle=True,
            random_state=self.random_state
        )

        records = []
        best_params_list = []

        for fold, (train_idx, test_idx) in enumerate(outer_split.split(X, y), start=1):
            # build fold DataFrames/Series
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            # inner loop hyperparameter tuning: get the best pipeline and its params
            best_pipe, best_params = self.model_tuning(model_key, X_train, y_train, inner_cv)
            best_params_list.append(best_params)

            # fit on the training fold
            best_pipe.fit(X_train, y_train)

            # predictions
            y_pred = best_pipe.predict(X_test)
            if hasattr(best_pipe, "predict_proba"):
                y_proba = best_pipe.predict_proba(X_test)[:, 1]
            else:
                y_proba = best_pipe.decision_function(X_test)

            # confusion-matrix stats
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            recall = recall_score(y_test, y_pred)
            specificity = tn / (tn + fp) if (tn + fp) else np.nan
            npv = tn / (tn + fn) if (tn + fn) else np.nan

            # Precision–Recall AUC
            prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc(rec_vals, prec_vals)

            # collect all metrics
            records.append({
                'fold': fold,
                'MCC': matthews_corrcoef(y_test, y_pred),
                'AUC': roc_auc_score(y_test, y_proba),
                'PRAUC': pr_auc,
                'BA': balanced_accuracy_score(y_test, y_pred),
                'F1': f1_score(y_test, y_pred),
                'F2': fbeta_score(y_test, y_pred, beta=2),
                'Recall': recall,
                'Specificity': specificity,
                'Precision': precision_score(y_test, y_pred),
                'NPV': npv
            })

        metrics_df = pd.DataFrame(records).set_index('fold')
        return metrics_df, best_params_list

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
            Number of outer fold loops.
        inner_cv : int
            Number of inner fold loops.
        num_rounds : int
            Number of rounds for the nCV.
        columns_to_remove : list, optional
            Columns to drop before CV.

        Returns
        -------
        dict
            {'metrics': DataFrame of num_rounds×outer_cv rows,
             'summary': DataFrame with median and 95% CI per metric,
             'best_params': list of param dicts per outer fold per round}
        """

        # collect per-round results and parameters
        all_df = []
        all_params = []

        # loop over the specified number of repeats
        for r in range(num_rounds):
            dfm, plist = self.outer_loop(
                df, target, model_key,
                outer_cv, inner_cv, columns_to_remove
            )
            # bring fold index into column before concatenation
            dfm = dfm.reset_index()
            # tag each row with the current repeat number (1-based)
            dfm['repeat'] = r + 1
            # append this round’s metrics and extend the parameter list
            all_df.append(dfm)
            all_params.extend(plist)

        # concatenate all rounds retaining both 'fold' and 'repeat'
        full = pd.concat(all_df, ignore_index=True)
        metrics = full.drop(columns=['repeat','fold'])

        # compute summary statistics
        summary = pd.DataFrame({
            'median': metrics.median(),
            'ci_lower': metrics.quantile(0.025),
            'ci_upper': metrics.quantile(0.975)
        })
        return {'metrics': full, 'summary': summary, 'best_params': all_params}


class Classifier:
    def __init__(self):
        self.model = LogisticRegression(penalty='elasticnet', solver='saga', random_state=0, max_iter=10000)

        self.param_grid = [{
      'penalty': ['elasticnet'],
      'C':       [0.01, 0.1, 1, 10],
      'l1_ratio':[0.0, 0.25, 0.5, 0.75, 1.0],
      'solver': ['saga'],
      'max_iter': [10000]   # saga can be slower; give it more room
    }
    ]

    def load_data(self, path):
        """
        Load data from the path were it was stored.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"The file at {path} was not found.")
        return pd.read_csv(path)

    def preprocess_data(self, df, columns_to_drop=None):
        """
        Remove unecessary columns, 
        encode categorical features,
        handle missing values and filling with column-wise 'mean'.
        """
        if columns_to_drop is None:
            columns_to_drop = []
        # drop unecessary columns 
        df=df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        # select columns that correspond to numerical features and make list
        num_list=df.select_dtypes(include=[np.number]).columns.tolist()
        # make list of all columns other than the columns that have numerical values -> categorical features list
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

    
    def train_tuned_model(self, model, X, y, cmap, scale=True):
        """
        Train model instance with hyperparameters.
        """
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
        y_pred = model.predict(X_test)

        # plot for actual vs predicted positives and negatives
        cm = confusion_matrix(y_test, y_pred)
        cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap=cmap)

        # compute AUC
        auc = roc_auc_score(y_test, y_pred)
        print(f"Model: {model.__class__.__name__} AUC: {auc:.4f}")

        return auc
    
    def train_final_model(self, model, X, y, save_path, scale=True):
        """
        Train model instance with hyperparameters.
        Save best model instance in provided path.
        """

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

    def evaluate_model(self, model, X, y, runs=30, test_size=0.2, scale=True, save_path=None):
        """
        Evaluate model instance. 
        """
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



