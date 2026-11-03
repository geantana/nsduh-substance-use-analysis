import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('34933-0001-Data.tsv', sep='\t', low_memory=False)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.3)
colors = sns.color_palette("husl", 10)

# Data preparation
df['cig_ever_binary'] = (df['CIGEVER'] == 1).astype(int)
df['mj_ever_binary'] = (df['MJEVER'] == 1).astype(int)
df['cig_init_age'] = df['CIGTRY'].apply(lambda x: x if x < 100 else np.nan)
df['mj_init_age'] = df['MJAGE'].apply(lambda x: x if x < 100 else np.nan)
df['age_group'] = df['CATAGE'].map({1: '12-17', 2: '18-25', 3: '26-34', 4: '35+'})
df['cig_risk_clean'] = df['RSKPKCIG'].apply(lambda x: x if x <= 4 else np.nan)
df['mj_risk_clean'] = df['RSKMJREG'].apply(lambda x: x if x <= 4 else np.nan)
df['mental_distress'] = np.nan
mask = (df['DSTNRV30'] <= 5) & (df['DSTHOP30'] <= 5) & (df['DSTRST30'] <= 5)
df.loc[mask, 'mental_distress'] = (6 - df.loc[mask, 'DSTNRV30']) + \
                                    (6 - df.loc[mask, 'DSTHOP30']) + \
                                    (6 - df.loc[mask, 'DSTRST30'])
df['cig_dependence'] = np.nan
dep_mask = (df['CIGIRTBL'] <= 5) & (df['CIGCRAVE'] <= 5)
df.loc[dep_mask, 'cig_dependence'] = df.loc[dep_mask, 'CIGIRTBL'] + df.loc[dep_mask, 'CIGCRAVE']

print("="*80)
print("GENERATING STATISTICAL ANALYSIS VISUALIZATIONS")
print("="*80)

# ============================================================================
# FIGURE 9: Prevalence by Age Group
# ============================================================================
print("\n[1/6] Creating Figure 9: Prevalence by Demographics...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Cigarette use by age
age_groups = ['12-17', '18-25', '26-34', '35+']
cig_prev = [df[df['age_group'] == age]['cig_ever_binary'].mean() * 100 for age in age_groups]
axes[0].bar(age_groups, cig_prev, color=colors[0], alpha=0.7, edgecolor='black')
axes[0].set_ylabel('Prevalence (%)', fontsize=12)
axes[0].set_xlabel('Age Group', fontsize=12)
axes[0].set_title('Cigarette Use Prevalence by Age', fontsize=14, fontweight='bold')
axes[0].set_ylim([0, 80])
for i, v in enumerate(cig_prev):
    axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Marijuana use by age
mj_prev = [df[df['age_group'] == age]['mj_ever_binary'].mean() * 100 for age in age_groups]
axes[1].bar(age_groups, mj_prev, color=colors[2], alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Prevalence (%)', fontsize=12)
axes[1].set_xlabel('Age Group', fontsize=12)
axes[1].set_title('Marijuana Use Prevalence by Age', fontsize=14, fontweight='bold')
axes[1].set_ylim([0, 80])
for i, v in enumerate(mj_prev):
    axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/fig9_prevalence_by_age.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/fig9_prevalence_by_age.png")

# ============================================================================
# FIGURE 10: Initiation Age by Gender
# ============================================================================
print("\n[2/6] Creating Figure 10: Initiation Age by Gender...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Cigarette initiation by gender
males_cig = df[df['IRSEX'] == 1]['cig_init_age'].dropna()
females_cig = df[df['IRSEX'] == 2]['cig_init_age'].dropna()

axes[0].hist(males_cig, bins=25, alpha=0.6, label='Males', color=colors[4], edgecolor='black')
axes[0].hist(females_cig, bins=25, alpha=0.6, label='Females', color=colors[5], edgecolor='black')
axes[0].axvline(males_cig.mean(), color=colors[4], linestyle='--', linewidth=2)
axes[0].axvline(females_cig.mean(), color=colors[5], linestyle='--', linewidth=2)
axes[0].set_xlabel('Age (years)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Cigarette Initiation Age by Gender', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Marijuana initiation by gender
males_mj = df[df['IRSEX'] == 1]['mj_init_age'].dropna()
females_mj = df[df['IRSEX'] == 2]['mj_init_age'].dropna()

axes[1].hist(males_mj, bins=25, alpha=0.6, label='Males', color=colors[4], edgecolor='black')
axes[1].hist(females_mj, bins=25, alpha=0.6, label='Females', color=colors[5], edgecolor='black')
axes[1].axvline(males_mj.mean(), color=colors[4], linestyle='--', linewidth=2)
axes[1].axvline(females_mj.mean(), color=colors[5], linestyle='--', linewidth=2)
axes[1].set_xlabel('Age (years)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Marijuana Initiation Age by Gender', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig10_initiation_gender.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/fig10_initiation_gender.png")

# ============================================================================
# FIGURE 11: Risk Perception vs Use
# ============================================================================
print("\n[3/6] Creating Figure 11: Risk Perception Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Cigarette risk perception by use status
risk_labels = ['No Risk', 'Slight', 'Moderate', 'Great']
cig_users_risk = df[df['cig_ever_binary'] == 1]['cig_risk_clean'].value_counts().sort_index()
cig_nonusers_risk = df[df['cig_ever_binary'] == 0]['cig_risk_clean'].value_counts().sort_index()

# Normalize to percentages
cig_users_pct = (cig_users_risk / cig_users_risk.sum() * 100).values
cig_nonusers_pct = (cig_nonusers_risk / cig_nonusers_risk.sum() * 100).values

x = np.arange(len(risk_labels))
width = 0.35

axes[0].bar(x - width/2, cig_users_pct, width, label='Users', color=colors[0], alpha=0.7, edgecolor='black')
axes[0].bar(x + width/2, cig_nonusers_pct, width, label='Non-users', color=colors[1], alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Risk Perception', fontsize=12)
axes[0].set_ylabel('Percentage (%)', fontsize=12)
axes[0].set_title('Cigarette Risk Perception by Use Status', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(risk_labels, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Marijuana risk perception by use status
mj_users_risk = df[df['mj_ever_binary'] == 1]['mj_risk_clean'].value_counts().sort_index()
mj_nonusers_risk = df[df['mj_ever_binary'] == 0]['mj_risk_clean'].value_counts().sort_index()

mj_users_pct = (mj_users_risk / mj_users_risk.sum() * 100).values
mj_nonusers_pct = (mj_nonusers_risk / mj_nonusers_risk.sum() * 100).values

axes[1].bar(x - width/2, mj_users_pct, width, label='Users', color=colors[2], alpha=0.7, edgecolor='black')
axes[1].bar(x + width/2, mj_nonusers_pct, width, label='Non-users', color=colors[3], alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Risk Perception', fontsize=12)
axes[1].set_ylabel('Percentage (%)', fontsize=12)
axes[1].set_title('Marijuana Risk Perception by Use Status', fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(risk_labels, rotation=45, ha='right')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/fig11_risk_perception_use.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/fig11_risk_perception_use.png")

# ============================================================================
# FIGURE 12: Frequency vs Dependence
# ============================================================================
print("\n[4/6] Creating Figure 12: Frequency and Dependence...")

freq_dep_data = df[['CIG30USE', 'cig_dependence']].dropna()
freq_dep_data = freq_dep_data[freq_dep_data['CIG30USE'] <= 30]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot
axes[0].scatter(freq_dep_data['CIG30USE'], freq_dep_data['cig_dependence'],
                alpha=0.3, color=colors[6], s=20)
# Add trend line
z = np.polyfit(freq_dep_data['CIG30USE'], freq_dep_data['cig_dependence'], 1)
p = np.poly1d(z)
r_freq_dep, p_freq_dep = stats.pearsonr(freq_dep_data['CIG30USE'], freq_dep_data['cig_dependence'])
p_label = 'p < 0.001' if p_freq_dep < 0.001 else f'p = {p_freq_dep:.3f}'
axes[0].plot(freq_dep_data['CIG30USE'].sort_values(),
             p(freq_dep_data['CIG30USE'].sort_values()),
             "r--", linewidth=2, label=f'r = {r_freq_dep:.3f}, {p_label}')
axes[0].set_xlabel('Days Smoked (Past 30 Days)', fontsize=12)
axes[0].set_ylabel('Dependence Score', fontsize=12)
axes[0].set_title('Smoking Frequency vs Dependence', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Box plot by frequency category
freq_dep_data['freq_category'] = pd.cut(freq_dep_data['CIG30USE'],
                                         bins=[0, 10, 20, 30],
                                         labels=['1-10 days', '11-20 days', '21-30 days'])
freq_dep_data.boxplot(column='cig_dependence', by='freq_category', ax=axes[1])
axes[1].set_xlabel('Smoking Frequency Category', fontsize=12)
axes[1].set_ylabel('Dependence Score', fontsize=12)
axes[1].set_title('Dependence by Smoking Frequency', fontsize=14, fontweight='bold')
axes[1].get_figure().suptitle('')  # Remove default title
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig12_frequency_dependence.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/fig12_frequency_dependence.png")

# ============================================================================
# FIGURE 13: Mental Health by Substance Use
# ============================================================================
print("\n[5/6] Creating Figure 13: Mental Health Analysis...")

mental_analysis = df[['cig_ever_binary', 'mj_ever_binary', 'mental_distress']].dropna()
mental_analysis['use_pattern'] = 'Neither'
mental_analysis.loc[(mental_analysis['cig_ever_binary']==1) & (mental_analysis['mj_ever_binary']==0), 'use_pattern'] = 'Cig Only'
mental_analysis.loc[(mental_analysis['cig_ever_binary']==0) & (mental_analysis['mj_ever_binary']==1), 'use_pattern'] = 'MJ Only'
mental_analysis.loc[(mental_analysis['cig_ever_binary']==1) & (mental_analysis['mj_ever_binary']==1), 'use_pattern'] = 'Both'

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Box plot
use_order = ['Neither', 'Cig Only', 'MJ Only', 'Both']
bp = axes[0].boxplot([mental_analysis[mental_analysis['use_pattern']==cat]['mental_distress']
                       for cat in use_order],
                      labels=use_order,
                      patch_artist=True)
for patch, color in zip(bp['boxes'], [colors[0], colors[1], colors[2], colors[3]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_xlabel('Substance Use Pattern', fontsize=12)
axes[0].set_ylabel('Mental Distress Score', fontsize=12)
axes[0].set_title('Mental Health by Substance Use Pattern', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Bar plot with means and error bars
means = [mental_analysis[mental_analysis['use_pattern']==cat]['mental_distress'].mean()
         for cat in use_order]
stds = [mental_analysis[mental_analysis['use_pattern']==cat]['mental_distress'].std()
        for cat in use_order]
axes[1].bar(use_order, means, yerr=stds, capsize=5,
            color=[colors[0], colors[1], colors[2], colors[3]],
            alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Substance Use Pattern', fontsize=12)
axes[1].set_ylabel('Mean Mental Distress Score', fontsize=12)
axes[1].set_title('Average Mental Distress by Use Pattern', fontsize=14, fontweight='bold')
for i, (m, s) in enumerate(zip(means, stds)):
    axes[1].text(i, m + s + 0.15, f'{m:.2f}', ha='center', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/fig13_mental_health.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/fig13_mental_health.png")

# ============================================================================
# FIGURE 14: Co-occurrence Analysis
# ============================================================================
print("\n[6/6] Creating Figure 14: Co-occurrence Patterns...")

fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Venn-like visualization
ax1 = fig.add_subplot(gs[0, :])
categories = ['Neither', 'Cig Only', 'MJ Only', 'Both']
contingency = pd.crosstab(df['cig_ever_binary'], df['mj_ever_binary'])  # rows: cig, cols: mj
n_neither, n_cig_only = contingency.loc[0, 0], contingency.loc[1, 0]
n_mj_only, n_both = contingency.loc[0, 1], contingency.loc[1, 1]
counts = [n_neither, n_cig_only, n_mj_only, n_both]
percentages = [c/len(df)*100 for c in counts]
colors_venn = [colors[7], colors[0], colors[2], colors[8]]

bars = ax1.bar(categories, percentages, color=colors_venn, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Percentage (%)', fontsize=12)
ax1.set_title('Co-occurrence Patterns of Cigarette and Marijuana Use', fontsize=14, fontweight='bold')
for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{pct:.1f}%\n(n={count:,})',
             ha='center', fontweight='bold', fontsize=11)
ax1.set_ylim([0, 50])
ax1.grid(True, alpha=0.3, axis='y')

# Heatmap
ax2 = fig.add_subplot(gs[1, 0])
contingency_pct = contingency / contingency.sum().sum() * 100
sns.heatmap(contingency_pct, annot=True, fmt='.1f', cmap='YlOrRd',
            xticklabels=['No MJ', 'Yes MJ'],
            yticklabels=['No Cig', 'Yes Cig'],
            cbar_kws={'label': 'Percentage (%)'},
            ax=ax2)
ax2.set_title('Co-occurrence Heatmap', fontsize=12, fontweight='bold')
ax2.set_xlabel('Marijuana Use', fontsize=11)
ax2.set_ylabel('Cigarette Use', fontsize=11)

# Conditional probabilities
ax3 = fig.add_subplot(gs[1, 1])
prob_mj_given_cig = n_both / (n_both + n_cig_only) * 100
prob_mj_given_no_cig = n_mj_only / (n_mj_only + n_neither) * 100
prob_cig_given_mj = n_both / (n_both + n_mj_only) * 100
prob_cig_given_no_mj = n_cig_only / (n_cig_only + n_neither) * 100

x_pos = [0, 1, 2.5, 3.5]
probs = [prob_mj_given_cig, prob_mj_given_no_cig, prob_cig_given_mj, prob_cig_given_no_mj]
labels = ['MJ|Cig', 'MJ|No Cig', 'Cig|MJ', 'Cig|No MJ']
bar_colors = [colors[2], colors[3], colors[0], colors[1]]

bars = ax3.bar(x_pos, probs, color=bar_colors, alpha=0.7, edgecolor='black')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(labels, rotation=0)
ax3.set_ylabel('Probability (%)', fontsize=11)
ax3.set_title('Conditional Probabilities', fontsize=12, fontweight='bold')
for bar, prob in zip(bars, probs):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{prob:.1f}%', ha='center', fontweight='bold')
ax3.set_ylim([0, 90])
ax3.grid(True, alpha=0.3, axis='y')

plt.savefig('figures/fig14_cooccurrence.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/fig14_cooccurrence.png")

print("\n" + "="*80)
print("ALL STATISTICAL VISUALIZATIONS COMPLETED!")
print("="*80)
print("\nGenerated 6 new figures:")
print("  9. fig9_prevalence_by_age.png")
print("  10. fig10_initiation_gender.png")
print("  11. fig11_risk_perception_use.png")
print("  12. fig12_frequency_dependence.png")
print("  13. fig13_mental_health.png")
print("  14. fig14_cooccurrence.png")
print("\nReady for complete Report 2 LaTeX generation!")