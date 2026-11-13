import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
import optuna

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
pd.set_option('display.max_columns', None)

raw_df = pd.read_excel(r'Submit_dataset.xlsx', sheet_name='stream')
df_monthly = raw_df.groupby(['site_id', 'year', 'month'], as_index=False).mean(numeric_only=True)

feature_ls = df_monthly.loc[:, 'CO2atm':'fForest'].columns.tolist()
feature_df = df_monthly[feature_ls]
label_col = 'DIC (mg/L)'
label_df = df_monthly[label_col]

X = df_monthly[feature_ls]
y = df_monthly[label_col].values
groups = df_monthly['site_id'].values

label_stratified = pd.cut(label_df.rank(method='first'), bins=10, labels=False)


def objective(trial):
    params = {
        'max_iter': trial.suggest_int('max_iter', 100, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 45),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 15, 60),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.01, 20.0, log=True),
        'max_bins': trial.suggest_int('max_bins', 128, 255)
    }
    optimizer = HistGradientBoostingRegressor(random_state=42, verbose=0, **params)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df_monthly))

    for train_idx, val_idx in skf.split(feature_df, label_stratified):
        X_train = feature_df.loc[train_idx, :]
        y_train = label_df.loc[train_idx]
        X_val = feature_df.loc[val_idx, :]

        optimizer.fit(X_train, y_train)
        oof_preds[val_idx] = optimizer.predict(X_val)

    return r2_score(label_df, oof_preds)


best_params = {'max_iter': 794, 'learning_rate': 0.07992772707809999, 'max_depth': 8, 'max_leaf_nodes': 38, 'min_samples_leaf': 32,
'l2_regularization': 9.930635343432396, 'max_bins': 225}

optimizer = HistGradientBoostingRegressor(random_state=42, verbose=0, **best_params)
optimizer.fit(feature_df, label_df)


import shap

explainer = shap.TreeExplainer(model=optimizer)
shap_values = explainer.shap_values(feature_df)
shap.summary_plot(shap_values, feature_df, plot_type='dot')
shap_df = pd.DataFrame(data=shap_values, columns=feature_df.columns.tolist())
