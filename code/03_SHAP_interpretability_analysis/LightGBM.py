import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.cluster import KMeans
from lightgbm import LGBMRegressor
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
y = df_monthly[label_col]
groups = df_monthly['site_id'].values

label_stratified = pd.cut(label_df.rank(method='first'), bins=10, labels=False)


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'num_leaves': trial.suggest_int('num_leaves', 10, 40),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'subsample_freq': 1,
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'min_child_samples': trial.suggest_int('min_child_samples', 15, 60),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
    }

    model = LGBMRegressor(**params, objective='regression', device='gpu', random_state=42, verbose=-1, gpu_device_id=0)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df_monthly))

    for train_idx, val_idx in skf.split(feature_df, label_stratified):
        X_train = feature_df.loc[train_idx, :]
        y_train = label_df.loc[train_idx]
        X_val = feature_df.loc[val_idx, :]

        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict(X_val)

    return r2_score(label_df, oof_preds)


best_params = {'n_estimators': 590, 'max_depth': 7, 'num_leaves': 31, 'learning_rate': 0.045383646844197226, 'subsample': 0.6004357984835966,
'colsample_bytree': 0.8679807796459401, 'min_child_samples': 17, 'reg_alpha': 0.13271553986071344, 'reg_lambda': 0.06452997894034963}

optimizer = LGBMRegressor(**best_params, objective='regression', device='gpu', random_state=42, verbose=-1, gpu_device_id=0)
optimizer.fit(feature_df, label_df)


import shap

explainer = shap.TreeExplainer(model=optimizer)
shap_values = explainer.shap_values(feature_df)
shap.summary_plot(shap_values, feature_df, plot_type='dot')
shap_df = pd.DataFrame(data=shap_values, columns=feature_df.columns.tolist())
