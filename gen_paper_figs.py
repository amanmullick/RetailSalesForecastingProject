import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 1. Model comparison bar chart (correct metrics) ──────────────────────────
metrics = {
    'Linear\nRegression': {'MAE': 2012.28, 'RMSE': 2772.64, 'R2': 0.2033, 'MAPE': 33.66},
    'Random\nForest':     {'MAE': 1034.71, 'RMSE': 1483.27, 'R2': 0.7720, 'MAPE': 16.54},
    'XGBoost':            {'MAE':  854.15, 'RMSE': 1198.13, 'R2': 0.8512, 'MAPE': 13.78},
}

models = list(metrics.keys())
colors = ['#5b9bd5', '#70ad47', '#ed7d31']   # blue, green, orange — readable in greyscale
x = np.arange(len(models))
bar_w = 0.55

fig, axes = plt.subplots(1, 4, figsize=(8.5, 2.4))

panel_cfg = [
    ('MAE (€)',   'MAE',  True,  [0, 2200]),
    ('RMSE (€)',  'RMSE', True,  [0, 3000]),
    ('R²',        'R2',   False, [0, 1.0]),
    ('MAPE (%)',  'MAPE', True,  [0, 38]),
]

for ax, (ylabel, key, lower_better, ylim) in zip(axes, panel_cfg):
    vals = [metrics[m][key] for m in models]
    bars = ax.bar(x, vals, width=bar_w, color=colors, edgecolor='white', linewidth=0.4)
    for bar, v in zip(bars, vals):
        label = f'{v:.2f}' if key != 'R2' else f'{v:.4f}'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ylim[1]*0.01,
                label, ha='center', va='bottom', fontsize=5.5, fontweight='bold')
    ax.set_ylim(ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=6.5)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(axis='y', labelsize=6.5)
    note = '↓ better' if lower_better else '↑ better'
    ax.set_title(note, fontsize=6, color='#555555', pad=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5)

plt.tight_layout(pad=0.8)
plt.savefig('paper/model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/model_comparison_corrected.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: paper/model_comparison.png")

# ── 2. CompetitionDistance SHAP dependence plot ───────────────────────────────
DATA    = 'data'
MODELS  = 'models'
FEATURE_COLS = [
    'Store','DayOfWeek','Promo','StateHoliday','SchoolHoliday',
    'StoreType','Assortment','CompetitionDistance',
    'CompetitionOpenSinceMonth','CompetitionOpenSinceYear',
    'Promo2','Promo2SinceWeek','Promo2SinceYear','PromoInterval',
    'Year','Month','Day','WeekOfYear','Quarter',
    'IsWeekend','IsMonthStart','IsMonthEnd',
    'CompetitionDuration','Promo2Duration'
]

train = pd.read_csv(f'{DATA}/train.csv', low_memory=False, parse_dates=['Date'])
store = pd.read_csv(f'{DATA}/store.csv')
df = train.merge(store, on='Store', how='left')
df = df[(df['Open'] == 1) & (df['Sales'] > 0)].copy()

df['Year']          = df['Date'].dt.year
df['Month']         = df['Date'].dt.month
df['Day']           = df['Date'].dt.day
df['WeekOfYear']    = df['Date'].dt.isocalendar().week.astype(int)
df['Quarter']       = df['Date'].dt.quarter
df['IsWeekend']     = (df['DayOfWeek'] >= 6).astype(int)
df['IsMonthStart']  = df['Date'].dt.is_month_start.astype(int)
df['IsMonthEnd']    = df['Date'].dt.is_month_end.astype(int)
df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
for col in ['CompetitionOpenSinceMonth','CompetitionOpenSinceYear',
            'Promo2SinceWeek','Promo2SinceYear']:
    df[col].fillna(0, inplace=True)
df['PromoInterval'].fillna('None', inplace=True)
df['CompetitionDuration'] = (
    (df['Year']  - df['CompetitionOpenSinceYear'])  * 12 +
    (df['Month'] - df['CompetitionOpenSinceMonth'])
).clip(lower=0)
df['Promo2Duration'] = (
    (df['Year']       - df['Promo2SinceYear']) * 52 +
    (df['WeekOfYear'] - df['Promo2SinceWeek'])
).clip(lower=0)
for col in ['StateHoliday','StoreType','Assortment','PromoInterval']:
    df[col] = df[col].astype('category').cat.codes

df.sample(frac=1, random_state=42).reset_index(drop=True, inplace=True)
split = int(len(df) * 0.8)
val   = df.iloc[split:].reset_index(drop=True)
X_val = val[FEATURE_COLS]

with open(f'{MODELS}/xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

X_shap   = X_val.sample(600, random_state=42).reset_index(drop=True)
explainer = shap.TreeExplainer(model)
shap_vals = explainer(X_shap)

# CompetitionDistance dependence plot
feat = 'CompetitionDistance'
feat_idx = FEATURE_COLS.index(feat)
feat_vals  = X_shap[feat].values
shap_cdist = shap_vals.values[:, feat_idx]

fig, ax = plt.subplots(figsize=(3.4, 2.6))
scatter = ax.scatter(feat_vals / 1000, shap_cdist,
                     c=X_shap['Assortment'].values,
                     cmap='RdBu_r', alpha=0.65, s=10, linewidths=0)
ax.axhline(0, color='#888', linewidth=0.6, linestyle='--')
ax.set_xlabel('Competition Distance (km)', fontsize=8)
ax.set_ylabel('SHAP value (€)', fontsize=8)
ax.tick_params(labelsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
cbar = plt.colorbar(scatter, ax=ax, pad=0.01, shrink=0.85)
cbar.set_label('Assortment', fontsize=7)
cbar.ax.tick_params(labelsize=6)
plt.tight_layout()
plt.savefig('paper/shap_dependence_cd.png', dpi=300, bbox_inches='tight')
plt.close()

# Print summary stats for CompetitionDistance SHAP
mask_close = feat_vals < 500
mask_far   = feat_vals > 15000
print(f"\nCompetitionDistance SHAP summary:")
print(f"  <500m  (n={mask_close.sum()}): mean SHAP = {shap_cdist[mask_close].mean():.1f} €")
print(f"  >15km  (n={mask_far.sum()}):  mean SHAP = {shap_cdist[mask_far].mean():.1f} €")

# Promo SHAP summary
promo_idx = FEATURE_COLS.index('Promo')
promo_vals = X_shap['Promo'].values
shap_promo = shap_vals.values[:, promo_idx]
print(f"\nPromo SHAP summary:")
print(f"  Promo=0 (n={(promo_vals==0).sum()}): mean SHAP = {shap_promo[promo_vals==0].mean():.1f} €")
print(f"  Promo=1 (n={(promo_vals==1).sum()}): mean SHAP = {shap_promo[promo_vals==1].mean():.1f} €")
print("Saved: paper/shap_dependence_cd.png")
