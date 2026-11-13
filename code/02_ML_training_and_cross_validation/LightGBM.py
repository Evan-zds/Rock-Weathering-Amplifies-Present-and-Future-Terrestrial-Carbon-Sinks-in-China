"""Random 10-fold cross-validation"""

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


# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=1e3)

best_params = {'n_estimators': 590, 'max_depth': 7, 'num_leaves': 31, 'learning_rate': 0.045383646844197226,
               'subsample': 0.6004357984835966,
               'colsample_bytree': 0.8679807796459401, 'min_child_samples': 17, 'reg_alpha': 0.13271553986071344,
               'reg_lambda': 0.06452997894034963}

optimizer = LGBMRegressor(**best_params, objective='regression', device='gpu', random_state=42, verbose=-1,
                          gpu_device_id=0)

# Cross-validation prediction
label_stratified = pd.cut(label_df.rank(method='first'), bins=10, labels=False)

folds = 10
skf = StratifiedKFold(n_splits=folds,
                      shuffle=True,
                      random_state=42)

assemble_val_preds, assemble_val_label = list(), list()

for k, (train_idx, val_idx) in enumerate(skf.split(feature_df, label_stratified)):
    optimizer.fit(feature_df.iloc[train_idx, :], label_df.iloc[train_idx])
    val_preds = optimizer.predict(feature_df.iloc[val_idx, :])

    assemble_val_preds.append(val_preds)
    assemble_val_label.append(label_df.iloc[val_idx])

    val_rmse = root_mean_squared_error(y_true=label_df.iloc[val_idx], y_pred=val_preds)
    val_r2 = r2_score(y_true=label_df.iloc[val_idx], y_pred=val_preds)

assemble_val_preds_array = np.concatenate(assemble_val_preds)  # y
assemble_val_label_array = np.concatenate(assemble_val_label).reshape(-1, )  # x


assemble_rmse = root_mean_squared_error(assemble_val_label_array, assemble_val_preds_array)
assemble_r2 = r2_score(assemble_val_label_array, assemble_val_preds_array)

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

pd.set_option('display.max_columns', None)

orig_cmap = plt.get_cmap("RdPu")
truncated_cmap = mpl.colors.ListedColormap(
    orig_cmap(np.linspace(0.175, 1.0, 256))
)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)

sns.scatterplot(x=assemble_val_label_array, y=assemble_val_preds_array, s=10, edgecolor='None',
                facecolor=truncated_cmap(0.0), legend=False, ax=ax, zorder=1)
sns.kdeplot(x=assemble_val_label_array, y=assemble_val_preds_array, fill=True, cmap=truncated_cmap, levels=100,
            thresh=0.4, bw_adjust=0.75, legend=False, ax=ax, zorder=2)
ax.plot(range(0, 151), range(0, 151), linestyle='--', linewidth=1.25, color='k')
ax.set_xlim(0, 150)
ax.set_ylim(0, 150)
ax.set_xticks([0, 50, 100, 150])
ax.set_yticks([0, 50, 100, 150])
ax.tick_params(direction='in', labelsize=14)
ax.set_xlabel('Observations (mg C L$^{-1}$)', fontsize=16)
ax.set_ylabel('Predictions (mg C L$^{-1}$)', fontsize=16)
ax.text(0.08, 0.925, 'Model = LightGBM', transform=ax.transAxes, ha='left', va='center', fontsize=14)
ax.text(0.08, 0.835,
        f'R$^2$ = {r2_score(assemble_val_label_array, assemble_val_preds_array):.2f}\nRMSE = {root_mean_squared_error(assemble_val_label_array, assemble_val_preds_array):.2f}',
        transform=ax.transAxes, ha='left', va='center', fontsize=14)
ax.text(0.88, 0.95, '1:1', transform=ax.transAxes, ha='left', va='center', fontsize=13)

plt.show()



"""site-level 10-fold cross-validation"""
best_params = {'n_estimators': 590, 'max_depth': 7, 'num_leaves': 31, 'learning_rate': 0.045383646844197226,
               'subsample': 0.6004357984835966,
               'colsample_bytree': 0.8679807796459401, 'min_child_samples': 17, 'reg_alpha': 0.13271553986071344,
               'reg_lambda': 0.06452997894034963}

model = LGBMRegressor(**best_params, objective='regression', device='gpu', random_state=42, verbose=-1, gpu_device_id=0)

n_splits = 10
gkf = GroupKFold(n_splits=n_splits)

oof_preds = np.zeros(len(df_monthly))

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val = X.iloc[val_idx]

    model.fit(X_train, y_train)

    oof_preds[val_idx] = np.clip(model.predict(X_val), 0, None)

df_monthly['pred_DIC'] = oof_preds
df_monthly['obs_DIC'] = y


r2_monthly = r2_score(df_monthly['obs_DIC'], df_monthly['pred_DIC'])
rmse_monthly = root_mean_squared_error(df_monthly['obs_DIC'], df_monthly['pred_DIC'])
print(f"[1] 逐月精度 (N = {len(df_monthly)} 个站点-月):")
print(f"    R2 = {r2_monthly:.3f}, RMSE = {rmse_monthly:.3f}")


df_annual = df_monthly.groupby(['site_id', 'year'], as_index=False)[['obs_DIC', 'pred_DIC']].mean()
r2_annual = r2_score(df_annual['obs_DIC'], df_annual['pred_DIC'])
rmse_annual = root_mean_squared_error(df_annual['obs_DIC'], df_annual['pred_DIC'])
print(f"\n[2] 逐年精度 (N = {len(df_annual)} 个站点-年):")
print(f"    R2 = {r2_annual:.3f}, RMSE = {rmse_annual:.3f}")


df_site = df_monthly.groupby('site_id', as_index=False)[['obs_DIC', 'pred_DIC']].mean()
r2_site = r2_score(df_site['obs_DIC'], df_site['pred_DIC'])
rmse_site = root_mean_squared_error(df_site['obs_DIC'], df_site['pred_DIC'])
print(f"\n[3] 空间变异 (N = {len(df_site)} 个独立站点):")
print(f"    R2 = {r2_site:.3f}, RMSE = {rmse_site:.3f}")


orig_cmap = plt.get_cmap("RdPu")
truncated_cmap = mpl.colors.ListedColormap(
    orig_cmap(np.linspace(0.175, 1.0, 256))
)

fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(nrows=1, ncols=3, wspace=0.25)

ax_site = fig.add_subplot(gs[0, 0])
ax_site.set_xlim(0, 150)
ax_site.set_ylim(0, 150)
ax_site.set_xticks([0, 50, 100, 150])
ax_site.set_yticks([0, 50, 100, 150])
ax_site.tick_params(axis='both', direction='in', labelsize=14)
sns.scatterplot(data=df_site, x='obs_DIC', y='pred_DIC', s=10, edgecolor='None', facecolor=truncated_cmap(0.0), legend=False, ax=ax_site, zorder=1)
sns.kdeplot(data=df_site, x='obs_DIC', y='pred_DIC', fill=True, cmap=truncated_cmap, levels=50, thresh=0.35, bw_adjust=0.65, legend=False, ax=ax_site, zorder=2)
# sns.regplot(data=df_site, x='obs_DIC', y='pred_DIC', ci=95, truncate=True, line_kws={'color': '#293890', 'linewidth': 1, 'zorder': 3}, ax=ax_site, scatter=False)
ax_site.plot(range(0, 151), range(0, 151), linestyle='--', linewidth=1.5, color='k')
ax_site.set_xlabel('Observations (mg C L$^{-1}$)', fontsize=16)
ax_site.set_ylabel('Predictions (mg C L$^{-1}$)', fontsize=16)
ax_site.set_title('Multi-year', fontsize=16, y=1.015)
ax_site.text(0.05, 0.905, f'R$^2$ = {r2_site:.2f}\nRMSE = {rmse_site:.2f}', transform=ax_site.transAxes, ha='left', va='center', fontsize=14)

ax_annual = fig.add_subplot(gs[0, 1])
ax_annual.set_xlim(0, 150)
ax_annual.set_ylim(0, 150)
ax_annual.set_xticks([0, 50, 100, 150])
ax_annual.set_yticks([0, 50, 100, 150])
ax_annual.set_yticklabels([])
ax_annual.tick_params(axis='both', direction='in', labelsize=14)
sns.scatterplot(data=df_annual, x='obs_DIC', y='pred_DIC', s=10, edgecolor='None', facecolor=truncated_cmap(0.0), legend=False,ax=ax_annual, zorder=1)
sns.kdeplot(data=df_annual, x='obs_DIC', y='pred_DIC', fill=True, cmap=truncated_cmap, levels=50, thresh=0.35, bw_adjust=0.65, legend=False,ax=ax_annual, zorder=2)
# sns.regplot(data=df_annual, x='obs_DIC', y='pred_DIC', ci=95, truncate=True, line_kws={'color': '#293890', 'linewidth': 1, 'zorder': 3},ax=ax_annual, scatter=False)
ax_annual.plot(range(0, 151), range(0, 151), linestyle='--', linewidth=1.5, color='k')
ax_annual.set_xlabel('Observations (mg C L$^{-1}$)', fontsize=16)
# ax_annual.set_ylabel('Predictions (mg L$^{-1}$)', fontsize=16)
ax_annual.set_ylabel('')
ax_annual.set_title('Yearly', fontsize=16, y=1.015)
ax_annual.text(0.05, 0.905, f'R$^2$ = {r2_annual:.2f}\nRMSE = {rmse_annual:.2f}', transform=ax_annual.transAxes, ha='left', va='center', fontsize=14)

ax_monthly = fig.add_subplot(gs[0, 2])
ax_monthly.set_xlim(0, 150)
ax_monthly.set_ylim(0, 150)
ax_monthly.set_xticks([0, 50, 100, 150])
ax_monthly.set_yticks([0, 50, 100, 150])
ax_monthly.set_yticklabels([])
ax_monthly.tick_params(axis='both', direction='in', labelsize=14)
sns.scatterplot(data=df_monthly, x='obs_DIC', y='pred_DIC', s=10, edgecolor='None', facecolor=truncated_cmap(0.0), legend=False,ax=ax_monthly, zorder=1)
sns.kdeplot(data=df_monthly, x='obs_DIC', y='pred_DIC', fill=True, cmap=truncated_cmap, levels=50, thresh=0.35, bw_adjust=0.6, legend=False,ax=ax_monthly, zorder=2)
# sns.regplot(data=df_monthly, x='obs_DIC', y='pred_DIC', ci=95, truncate=True, line_kws={'color': '#293890', 'linewidth': 1, 'zorder': 3},ax=ax_monthly, scatter=False)
ax_monthly.plot(range(0, 151), range(0, 151), linestyle='--', linewidth=1.5, color='k')
ax_monthly.set_xlabel('Observations (mg C L$^{-1}$)', fontsize=16)
# ax_annual.set_ylabel('Predictions (mg L$^{-1}$)', fontsize=16)
ax_monthly.set_ylabel('')
ax_monthly.set_title('Monthly', fontsize=16, y=1.015)
ax_monthly.text(0.05, 0.905, f'R$^2$ = {r2_monthly:.2f}\nRMSE = {rmse_monthly:.2f}', transform=ax_monthly.transAxes, ha='left', va='center', fontsize=14)

cax = fig.add_axes([0.95, 0.11, 0.0225, .775])
sm = mpl.cm.ScalarMappable(
    cmap=truncated_cmap, norm=mpl.colors.Normalize(vmin=0, vmax=1)
)
sm.set_array([])

cb = fig.colorbar(sm, cax=cax)
cb.set_label("Density", fontsize=16, labelpad=10, fontweight="regular", rotation=-90)
cb.set_ticks([0, 1])
cb.set_ticklabels(["Low", "High"])
cb.ax.tick_params(labelsize=14, length=0, pad=8)

plt.show()