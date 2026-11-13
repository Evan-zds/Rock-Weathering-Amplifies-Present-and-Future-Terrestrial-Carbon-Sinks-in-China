"""Hydrochemical reconstruction"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


pd.set_option('display.max_columns', None)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

Twater_df = pd.read_excel(r'Submit_dataset.xlsx').loc[:, ['Tsoilsur', 'Twater (℃)']]
pHwater_df = pd.read_excel(r'Submit_dataset.xlsx').loc[:, ['pH', 'pHsur', 'pHsub']]


"""TWater"""
fig = plt.figure(figsize=(11, 3))
gs = fig.add_gridspec(nrows=1, ncols=2, wspace=0.5)
ax0 = fig.add_subplot(gs[0, 0])

clean_T = Twater_df[['Tsoilsur', 'Twater (℃)']].dropna()
x_T = clean_T['Tsoilsur'].values
y_T = clean_T['Twater (℃)'].values
n_T = len(x_T)

res_T = stats.linregress(x_T, y_T)
slope_T, intercept_T = res_T.slope, res_T.intercept
r2_T = res_T.rvalue**2
p_T = res_T.pvalue

y_pred_T = slope_T * x_T + intercept_T
sigma_res_T = np.sqrt(np.sum((y_T - y_pred_T)**2) / (n_T - 2))
x_line_T = np.linspace(x_T.min(), x_T.max(), 200)
y_line_T = slope_T * x_line_T + intercept_T

t_val_T = 1.0

sxx_T = np.sum((x_T - np.mean(x_T))**2)
pi_se_T = sigma_res_T * np.sqrt(1 + 1.0/n_T + (x_line_T - np.mean(x_T))**2 / sxx_T)
y_pi_upper_T = y_line_T + t_val_T * pi_se_T
y_pi_lower_T = y_line_T - t_val_T * pi_se_T

sns.scatterplot(x=x_T, y=y_T, ax=ax0, edgecolor='white', s=45, color='#67b1d7', alpha=0.75, zorder=2)
ax0.plot([-20, 40], [-20, 40], linewidth=1.5, color='k', linestyle='--', zorder=5) # 1:1对角线
ax0.fill_between(x_line_T, y_pi_lower_T, y_pi_upper_T, color='red', alpha=0.1, zorder=3, label='95% Prediction Interval')
ax0.plot(x_line_T, y_line_T, color='red', linewidth=2, zorder=4)
ax0.set_xlim(-20, 40)
ax0.set_xticks([-20, 0, 20, 40])
ax0.set_ylim(-20, 40)
ax0.set_yticks([-20, 0, 20, 40])
ax0.tick_params(axis='both', which='both', labelsize=17, direction='in')
ax0.set_xlabel('T$_{\mathrm{sur}}$ (°C)', fontsize=20)
ax0.set_ylabel('T$_{\mathrm{water}}$ (°C)', fontsize=20)

p_text_T = "$p$ < 0.001" if p_T < 0.001 else f"p = {p_T:.3f}"
ax0.text(0.06, 0.94,
         f'R$^2$ = {r2_T:.2f}\n'
         f'σ = {sigma_res_T:.2f}\n'
         # f'BIAS = {}\n'
         f'N = {n_T:,}\n'
         f'{p_text_T}',
         transform=ax0.transAxes, ha='left', va='top', fontsize=14)
sign_T = '+' if intercept_T >= 0 else '-'
eq_text_T = f'$\mathrm{{T}}_{{\mathrm{{water}}}} = {slope_T:.2f} × \mathrm{{T}}_{{\mathrm{{sur}}}} {sign_T} {abs(intercept_T):.2f}$'
ax0.text(0.95, 0.06, eq_text_T,
         transform=ax0.transAxes, ha='right', va='bottom', fontsize=14, color='k')

pos_ax = ax0.get_position()
left_ax, bottom_ax, width_ax, height_ax = pos_ax.bounds



"""pHWater"""
ax1 = fig.add_subplot(gs[0, 1])
pHwater_df['pHMid'] = 30/100 * pHwater_df['pHsur'] + 70/100 * pHwater_df['pHsub']
pHwater_df['pHDelta'] = pHwater_df['pH'] - pHwater_df['pHmid']
clean_pH = pHwater_df[['pHMid', 'pHDelta']].dropna()
x_pH = clean_pH['pHMid'].values
y_pH = clean_pH['pHDelta'].values
n_pH = len(x_pH)

res_pH = stats.linregress(x_pH, y_pH)
slope_pH, intercept_pH = res_pH.slope, res_pH.intercept
r2_pH = res_pH.rvalue**2
p_pH = res_pH.pvalue

y_pred_pH = slope_pH * x_pH + intercept_pH
sigma_res_pH = np.sqrt(np.sum((y_pH - y_pred_pH)**2) / (n_pH - 2))
x_line_pH = np.linspace(x_pH.min(), x_pH.max(), 200)
y_line_pH = slope_pH * x_line_pH + intercept_pH

t_val_pH = 1.0

sxx_pH = np.sum((x_pH - np.mean(x_pH))**2)
pi_se_pH = sigma_res_pH * np.sqrt(1 + 1.0/n_pH + (x_line_pH - np.mean(x_pH))**2 / sxx_pH)
y_pi_upper_pH = y_line_pH + t_val_pH * pi_se_pH
y_pi_lower_pH = y_line_pH - t_val_pH * pi_se_pH

sns.scatterplot(x=x_pH, y=y_pH, ax=ax1, edgecolor='white', s=45, color='#67b1d7', alpha=0.75, zorder=2)
ax1.axhline(0, linewidth=1.5, color='k', linestyle='--', zorder=5) # 0基准线
ax1.fill_between(x_line_pH, y_pi_lower_pH, y_pi_upper_pH, color='red', alpha=0.1, zorder=3, label='95% Prediction Interval')
ax1.plot(x_line_pH, y_line_pH, color='red', linewidth=2, zorder=4)

ax1.set_xlim(5, 9)
ax1.set_xticks([5, 6, 7, 8, 9])
ax1.set_ylim(-2, 6)
ax1.set_yticks([-2, 0, 2, 4, 6])
ax1.tick_params(axis='both', which='both', labelsize=17, direction='in')
ax1.set_xlabel('pH$_{\mathrm{soil}}$', fontsize=20)
ax1.set_ylabel('Δ pH', fontsize=20)

p_text_pH = "$p$ < 0.001" if p_pH < 0.001 else f"p = {p_pH:.3f}"
ax1.text(0.6, 0.94,
         f'R$^2$ = {r2_pH:.2f}\n'
         f'σ = {sigma_res_pH:.2f}\n'
         f'N = {n_pH:,}\n'
         f'{p_text_pH}',
         transform=ax1.transAxes, ha='left', va='top', fontsize=14)
sign_pH = '+' if intercept_pH >= 0 else '-'
eq_text_pH = f'$\Delta\mathrm{{pH}} = {slope_pH:.2f} × \mathrm{{pH}}_{{\mathrm{{soil}}}} {sign_pH} {abs(intercept_pH):.2f}$'
ax1.text(0.06, 0.06, eq_text_pH,
         transform=ax1.transAxes, ha='left', va='bottom', fontsize=14, color='k')
ax1.set_title('ΔpH = pH$_{\mathrm{water}}$ − pH$_{\mathrm{soil}}$', fontsize=16)



ax0.text(-0.3, 0.985, '(A)', transform=ax0.transAxes, fontsize=21)
ax1.text(-0.25, 0.985, '(B)', transform=ax1.transAxes, fontsize=21)

plt.show()


"""Carbonate equilibrium"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib as mpl
import PyCO2SYS


pd.set_option('display.max_columns', None)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'


df = pd.read_excel(r"Submit_dataset.xlsx")



fig = plt.figure(figsize=(8, 6))
ax2 = fig.add_subplot(111)

df['DIC (umol/L)'] = df['DIC (mg/L)'] / 12 * 1000
df['DIC_calculate (umol/L)'] = df['DIC_calculate (mg/L)'] / 12 * 1000

types_config = [
    ('DIC', 'DIC (umol/L)', 'DIC_calculate (umol/L)', '#e38d8c',
     r'DIC'),
    ('HCO3-', 'HCO3- (umol/L)', 'HCO3_calculate (umol/L)', '#f2b74d',
     r'HCO$_3^-$'),
    ('Alkalinity', 'Alk (umol/L)', 'Alk_calculate (umol/L)', '#67b1d7',
     r'Alkalinity')
]

stat_summary = []
dfs = []

for key, obs_col, calc_col, color, label_tex in types_config:
    sub_df = df[[obs_col, calc_col]].dropna().reset_index(drop=True)
    sub_df.columns = ['Observation', 'Calculation']
    sub_df['type'] = key
    dfs.append(sub_df)

    x_val = sub_df['Observation'].values
    y_val = sub_df['Calculation'].values
    n_val = len(x_val)

    res = stats.linregress(x_val, y_val)
    rmse_val = np.sqrt(np.mean((y_val - (res.slope * x_val + res.intercept)) ** 2))

    stat_summary.append({
        'key': key,
        'label_tex': label_tex,
        'color': color,
        'k': res.slope,
        'r2': res.rvalue ** 2,
        'rmse': rmse_val,
        'n': n_val
    })

plot_df = pd.concat(dfs, axis=0)
palette_dict = {item['key']: item['color'] for item in stat_summary}

sns.scatterplot(
    data=plot_df, x='Observation', y='Calculation', hue='type',
    palette=palette_dict, s=65, alpha=0.75, edgecolor='white', linewidth=0.6,
    ax=ax2, zorder=3
)

ax2.plot([0, 7500], [0, 7500], linewidth=1.5, color='k', linestyle='--', zorder=2)

ax2.set_xlim(0, 7500)
ax2.set_xticks([0, 2500, 5000, 7500])
ax2.set_ylim(0, 7500)
ax2.set_yticks([0, 2500, 5000, 7500])
ax2.tick_params(axis='both', which='both', labelsize=14, direction='in')
ax2.set_xlabel('Observations (μmol L$^{\mathrm{-1}}$)', fontsize=17)
ax2.set_ylabel('Calculations (μmol L$^{\mathrm{-1}}$)', fontsize=17)

if ax2.get_legend():
    ax2.get_legend().remove()

legend_handles = [
    mpl.lines.Line2D(
        [0],
        [0],
        marker='o',
        color='none',  # 不画线
        markerfacecolor=item['color'],  # 与散点同色
        markeredgecolor='white',  # 白色细边框
        markeredgewidth=0.8,
        markersize=9,  # 圆圈大小
        linestyle='',
    )
    for item in stat_summary
]

legend_labels = [
    (
        f"{item['label_tex']}:   R$^2$ = {item['r2']:.2f}   RMSE ="
        f" {item['rmse']:.1f}   N = {item['n']:,}"
    )
    for item in stat_summary
]

leg = ax2.legend(
    handles=legend_handles,
    labels=legend_labels,
    loc='upper left',
    bbox_to_anchor=(-0.025, 1.02),  # 调节左上角相对位置
    frameon=False,  # 无边框背景
    fontsize=14,  # 字号与原文字一致
    handletextpad=0.5,  # 圆圈与文字的水平间距
    labelspacing=0.15,  # 每行之间的垂直间距
)

plt.show()