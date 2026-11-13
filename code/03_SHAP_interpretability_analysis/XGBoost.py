import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
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
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),  # XGBoost 树通常较浅，3-10 即可
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),  # 梯度提升的核心学习率
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # 行采样比例（类似RF的Bootstrap）
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # 列采样比例（对应RF的 max_features）
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  # 类似叶子节点最小样本权重
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),  # L1正则化
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),  # L2正则化
    }

    model = XGBRegressor(**params, booster='gbtree', objective='reg:squarederror', tree_method='hist',
                         device='cuda:0', random_state=42)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df_monthly))

    for train_idx, val_idx in skf.split(feature_df, label_stratified):
        X_train = feature_df.loc[train_idx, :]
        y_train = label_df.loc[train_idx]
        X_val = feature_df.loc[val_idx, :]
        y_val = label_df.loc[val_idx]

        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict(X_val)

    return r2_score(label_df, oof_preds)


best_params = {'n_estimators': 538, 'max_depth': 10, 'learning_rate': 0.020266007580476524, 'subsample': 0.6422345667575905,
'colsample_bytree': 0.8017867311735984, 'min_child_weight': 5, 'reg_alpha': 0.0002079475552171627, 'reg_lambda': 8.036177050073895e-05}

optimizer = XGBRegressor(random_state=42, booster='gbtree', objective='reg:squarederror', tree_method="hist",
                         device="cuda:0", **best_params)
optimizer.fit(feature_df, label_df)


import shap

explainer = shap.TreeExplainer(model=optimizer)
shap_values = explainer.shap_values(feature_df)
shap.summary_plot(shap_values, feature_df, plot_type='dot')
shap_df = pd.DataFrame(data=shap_values, columns=feature_df.columns.tolist())
