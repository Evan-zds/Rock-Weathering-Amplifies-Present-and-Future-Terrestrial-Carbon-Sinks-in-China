import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.cluster import KMeans
from catboost import CatBoostRegressor
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
        'iterations': trial.suggest_int('iterations', 200, 1000),
        'depth': trial.suggest_int('depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 30.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 15, 60),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10.0, log=True)
    }

    optimizer = CatBoostRegressor(
        **params,
        bootstrap_type='Poisson',
        loss_function='RMSE',
        task_type="GPU",
        devices='0',
        random_state=42,
        verbose=0
    )

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df_monthly))

    for train_idx, val_idx in skf.split(feature_df, label_stratified):
        X_train = feature_df.loc[train_idx, :]
        y_train = label_df.loc[train_idx]
        X_val = feature_df.loc[val_idx, :]

        optimizer.fit(X_train, y_train)
        oof_preds[val_idx] = optimizer.predict(X_val)

    return r2_score(label_df, oof_preds)

best_params = {'iterations': 978, 'depth': 8, 'learning_rate': 0.02841927464540473, 'l2_leaf_reg': 5.845420577455735,
'subsample': 0.9024744934818496, 'min_data_in_leaf': 53, 'random_strength': 0.526551012551942}

optimizer = CatBoostRegressor(
        **best_params,
        bootstrap_type='Poisson',
        loss_function='RMSE',
        task_type="GPU",
        devices='0',
        random_state=42,
        verbose=0)
optimizer.fit(feature_df, label_df)


import shap

explainer = shap.TreeExplainer(model=optimizer)
shap_values = explainer.shap_values(feature_df)
shap.summary_plot(shap_values, feature_df, plot_type='dot')
shap_df = pd.DataFrame(data=shap_values, columns=feature_df.columns.tolist())
