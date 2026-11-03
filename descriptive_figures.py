import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load data (only the columns used below)
cols = ['CIGEVER', 'MJEVER', 'CIGTRY', 'MJAGE', 'CIG30USE', 'MRDAYPWK',
        'RSKPKCIG', 'RSKMJREG', 'RSKDIFMJ', 'RKFQDNGR', 'CIGIRTBL', 'CIGCRAVE',
        'DSTNRV30', 'DSTHOP30', 'DSTRST30', 'CATAGE', 'SEXAGE', 'MMGETMJ']
df = pd.read_csv('34933-0001-Data.tsv', sep='\t', usecols=cols, low_memory=False)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.3)
colors = sns.color_palette("husl", 10)
os.makedirs('figures', exist_ok=True)


def valid_counts(series, codes):
    """Counts for the given valid codes only (NSDUH missing/skip codes are dropped)."""
    return series.value_counts().reindex(codes, fill_value=0)


def bar_panel(ax, series, codes, labels, title, color, rotate=True):
    counts = valid_counts(series, codes)
    ax.bar(labels, counts.values, color=color, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')


print("="*80)
print("GENERATING DESCRIPTIVE VISUALIZATIONS (Figures 1-8)")
print("="*80)

# ============================================================================
# FIGURE 1: Lifetime Prevalence
# ============================================================================
print("\n[1/8] Creating Figure 1: Lifetime Prevalence...")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
for ax, var, title, cols_pie in [(axes[0], 'CIGEVER', 'Ever Smoked Cigarettes', [colors[0], colors[1]]),
                                 (axes[1], 'MJEVER', 'Ever Used Marijuana', [colors[2], colors[3]])]:
    yes = (df[var] == 1).sum()
    ax.pie([yes, len(df) - yes], labels=['Yes', 'No'], autopct='%1.1f%%',
           colors=cols_pie, startangle=90)
    ax.set_title(title, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/fig1_prevalence.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/fig1_prevalence.png")

# ============================================================================
# FIGURE 2: Age of Initiation
# ============================================================================
print("\n[2/8] Creating Figure 2: Age of Initiation...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, var, title, color in [(axes[0], 'CIGTRY', 'Age First Smoked Cigarette', colors[0]),
                              (axes[1], 'MJAGE', 'Age First Used Marijuana', colors[2])]:
    ages = df[var][df[var] < 100]  # >= 985 are missing/skip codes
    ax.hist(ages, bins=30, color=color, alpha=0.7, edgecolor='black')
    ax.axvline(ages.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ages.mean():.1f}')
    ax.axvline(ages.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {ages.median():.1f}')
    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig2_age_initiation.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/fig2_age_initiation.png")

# ============================================================================
# FIGURE 3: Recent Use
# ============================================================================
print("\n[3/8] Creating Figure 3: Recent Use Patterns...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cig30 = df['CIG30USE'][df['CIG30USE'] <= 30]
axes[0].hist(cig30, bins=30, color=colors[0], alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Days Used', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Cigarette Use in Past 30 Days', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

mj_week = valid_counts(df['MRDAYPWK'], range(1, 8))
axes[1].bar(mj_week.index, mj_week.values, color=colors[2], alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Days per Week', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Marijuana Use: Days per Week', fontsize=14, fontweight='bold')
axes[1].set_xticks(range(1, 8))
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/fig3_recent_use.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/fig3_recent_use.png")

# ============================================================================
# FIGURE 4: Risk Perception
# ============================================================================
print("\n[4/8] Creating Figure 4: Risk Perception...")

risk_labels = ['No Risk', 'Slight Risk', 'Moderate Risk', 'Great Risk']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
bar_panel(axes[0, 0], df['RSKPKCIG'], [1, 2, 3, 4], risk_labels, 'Risk: Smoking 1+ Pack/Day', colors[0])
bar_panel(axes[0, 1], df['RSKMJREG'], [1, 2, 3, 4], risk_labels, 'Risk: Using MJ 1-2x/Week', colors[2])
bar_panel(axes[1, 0], df['RSKDIFMJ'], [1, 2, 3, 4, 5],
          ['Probably\nImpossible', 'Very Difficult', 'Fairly Difficult', 'Fairly Easy', 'Very Easy'],
          'Difficulty Getting Marijuana', colors[4])
bar_panel(axes[1, 1], df['RKFQDNGR'], [1, 2, 3, 4],
          ['Never', 'Seldom', 'Sometimes', 'Always'],
          'Get a Kick out of Dangerous Things', colors[5])

plt.tight_layout()
plt.savefig('figures/fig4_risk_perception.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/fig4_risk_perception.png")

# ============================================================================
# FIGURE 5: Dependence Symptoms
# ============================================================================
print("\n[5/8] Creating Figure 5: Dependence Symptoms...")

dep_labels = ['Not at All\nTrue', 'Somewhat\nTrue', 'Moderately\nTrue', 'Very True', 'Extremely\nTrue']
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
bar_panel(axes[0], df['CIGIRTBL'], [1, 2, 3, 4, 5], dep_labels, 'Need Cigarettes to Feel Less Irritable', colors[6])
bar_panel(axes[1], df['CIGCRAVE'], [1, 2, 3, 4, 5], dep_labels, 'Crave Cigarettes After Few Hours', colors[7])

plt.tight_layout()
plt.savefig('figures/fig5_dependence.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/fig5_dependence.png")

# ============================================================================
# FIGURE 6: Mental Health Indicators (adults only; youths are code 99 = legitimate skip)
# ============================================================================
print("\n[6/8] Creating Figure 6: Mental Health Indicators...")

mh_labels = ['All of\nthe Time', 'Most of\nthe Time', 'Some of\nthe Time', 'A Little of\nthe Time', 'None of\nthe Time']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
bar_panel(axes[0], df['DSTNRV30'], [1, 2, 3, 4, 5], mh_labels, 'Felt Nervous (Past 30 Days)', colors[8])
bar_panel(axes[1], df['DSTHOP30'], [1, 2, 3, 4, 5], mh_labels, 'Felt Hopeless (Past 30 Days)', colors[9])
bar_panel(axes[2], df['DSTRST30'], [1, 2, 3, 4, 5], mh_labels, 'Felt Restless (Past 30 Days)', colors[0])

plt.tight_layout()
plt.savefig('figures/fig6_mental_health.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/fig6_mental_health.png")

# ============================================================================
# FIGURE 7: Demographics
# ============================================================================
print("\n[7/8] Creating Figure 7: Demographics...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
bar_panel(axes[0], df['CATAGE'], [1, 2, 3, 4], ['12-17', '18-25', '26-34', '35+'],
          'Age Category Distribution', colors[1], rotate=False)
bar_panel(axes[1], df['SEXAGE'], [1, 2, 3, 4, 5],
          ['Male 12-17', 'Female 12-17', 'Male 18-25', 'Female 18-25', 'Aged 26+\n(both)'],
          'Gender by Age Category', colors[3])

plt.tight_layout()
plt.savefig('figures/fig7_demographics.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/fig7_demographics.png")

# ============================================================================
# FIGURE 8: Marijuana Acquisition
# ============================================================================
print("\n[8/8] Creating Figure 8: Marijuana Acquisition...")

fig, ax = plt.subplots(figsize=(10, 6))
bar_panel(ax, df['MMGETMJ'], [1, 2, 3, 4], ['Bought', 'Traded', 'Shared/Given', 'Grew'],
          'How Last Marijuana Was Obtained', colors[6], rotate=False)
ax.title.set_fontsize(14)

plt.tight_layout()
plt.savefig('figures/fig8_mj_acquisition.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/fig8_mj_acquisition.png")

print("\n" + "="*80)
print("ALL DESCRIPTIVE VISUALIZATIONS COMPLETED!")
print("="*80)
