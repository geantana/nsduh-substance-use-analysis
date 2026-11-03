import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('34933-0001-Data.tsv', sep='\t', low_memory=False)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
colors = sns.color_palette("husl", 10)

print("="*80)
print("REGRESSION ANALYSIS - PREDICTIVE MODELING")
print("="*80)

# ============================================================================
# DATA PREPARATION
# ============================================================================
print("\n[STEP 1] Data Preparation for Regression Models...")
print("-"*80)

# Create binary outcomes
df['cig_ever'] = (df['CIGEVER'] == 1).astype(int)
df['mj_ever'] = (df['MJEVER'] == 1).astype(int)

# Clean age variables
df['cig_init_age'] = df['CIGTRY'].apply(lambda x: x if x < 100 else np.nan)
df['mj_init_age'] = df['MJAGE'].apply(lambda x: x if x < 100 else np.nan)

# Demographics
df['age_cat'] = df['CATAGE']  # 1=12-17, 2=18-25, 3=26-34, 4=35+
df['gender'] = df['IRSEX']  # 1=Male, 2=Female

# Risk perceptions
df['cig_risk'] = df['RSKPKCIG'].apply(lambda x: x if x <= 4 else np.nan)
df['mj_risk'] = df['RSKMJREG'].apply(lambda x: x if x <= 4 else np.nan)

# Mental health composite
mask = (df['DSTNRV30'] <= 5) & (df['DSTHOP30'] <= 5) & (df['DSTRST30'] <= 5)
df.loc[mask, 'mental_distress'] = (6 - df.loc[mask, 'DSTNRV30']) + \
                                    (6 - df.loc[mask, 'DSTHOP30']) + \
                                    (6 - df.loc[mask, 'DSTRST30'])

print(f"Total sample: {len(df):,}")
print(f"Cigarette users: {df['cig_ever'].sum():,} ({df['cig_ever'].mean()*100:.1f}%)")
print(f"Marijuana users: {df['mj_ever'].sum():,} ({df['mj_ever'].mean()*100:.1f}%)")

# ============================================================================
# MODEL 1: LOGISTIC REGRESSION - CIGARETTE USE
# ============================================================================
print("\n" + "="*80)
print("MODEL 1: LOGISTIC REGRESSION - PREDICTING CIGARETTE USE")
print("="*80)

# Select features
features_cig = ['age_cat', 'gender', 'cig_risk', 'mental_distress']
X_cig = df[features_cig].dropna()
y_cig = df.loc[X_cig.index, 'cig_ever']

print(f"\nFeatures: {features_cig}")
print(f"Sample size after removing missing: {len(X_cig):,}")
print(f"Class distribution: No={sum(y_cig==0)} ({sum(y_cig==0)/len(y_cig)*100:.1f}%), Yes={sum(y_cig==1)} ({sum(y_cig==1)/len(y_cig)*100:.1f}%)")

# Split data
X_train_cig, X_test_cig, y_train_cig, y_test_cig = train_test_split(
    X_cig, y_cig, test_size=0.2, random_state=42, stratify=y_cig
)

print(f"\nTrain set: {len(X_train_cig):,}")
print(f"Test set: {len(X_test_cig):,}")

# Standardize features
scaler_cig = StandardScaler()
X_train_cig_scaled = scaler_cig.fit_transform(X_train_cig)
X_test_cig_scaled = scaler_cig.transform(X_test_cig)

# Train logistic regression
lr_cig = LogisticRegression(max_iter=1000, random_state=42)
lr_cig.fit(X_train_cig_scaled, y_train_cig)

# Predictions
y_pred_cig = lr_cig.predict(X_test_cig_scaled)
y_pred_proba_cig = lr_cig.predict_proba(X_test_cig_scaled)[:, 1]

# Performance metrics
print("\n" + "-"*80)
print("CIGARETTE MODEL PERFORMANCE:")
print("-"*80)

accuracy_cig = accuracy_score(y_test_cig, y_pred_cig)
precision_cig = precision_score(y_test_cig, y_pred_cig)
recall_cig = recall_score(y_test_cig, y_pred_cig)
f1_cig = f1_score(y_test_cig, y_pred_cig)
roc_auc_cig = roc_auc_score(y_test_cig, y_pred_proba_cig)

print(f"Accuracy:  {accuracy_cig:.4f}")
print(f"Precision: {precision_cig:.4f}")
print(f"Recall:    {recall_cig:.4f}")
print(f"F1-Score:  {f1_cig:.4f}")
print(f"ROC-AUC:   {roc_auc_cig:.4f}")

# Confusion matrix
cm_cig = confusion_matrix(y_test_cig, y_pred_cig)
print(f"\nConfusion Matrix:")
print(f"                Predicted")
print(f"                No    Yes")
print(f"Actual No    {cm_cig[0,0]:5d} {cm_cig[0,1]:5d}")
print(f"Actual Yes   {cm_cig[1,0]:5d} {cm_cig[1,1]:5d}")

# Coefficients
print(f"\nModel Coefficients (Log-Odds):")
for i, feature in enumerate(features_cig):
    coef = lr_cig.coef_[0][i]
    odds_ratio = np.exp(coef)
    print(f"  {feature:20s}: β={coef:7.4f}, OR={odds_ratio:7.4f}")
print(f"  {'Intercept':20s}: β={lr_cig.intercept_[0]:7.4f}")

# ============================================================================
# MODEL 2: LOGISTIC REGRESSION - MARIJUANA USE
# ============================================================================
print("\n" + "="*80)
print("MODEL 2: LOGISTIC REGRESSION - PREDICTING MARIJUANA USE")
print("="*80)

# Select features
features_mj = ['age_cat', 'gender', 'mj_risk', 'mental_distress', 'cig_ever']
X_mj = df[features_mj].dropna()
y_mj = df.loc[X_mj.index, 'mj_ever']

print(f"\nFeatures: {features_mj}")
print(f"Sample size after removing missing: {len(X_mj):,}")
print(f"Class distribution: No={sum(y_mj==0)} ({sum(y_mj==0)/len(y_mj)*100:.1f}%), Yes={sum(y_mj==1)} ({sum(y_mj==1)/len(y_mj)*100:.1f}%)")

# Split data
X_train_mj, X_test_mj, y_train_mj, y_test_mj = train_test_split(
    X_mj, y_mj, test_size=0.2, random_state=42, stratify=y_mj
)

print(f"\nTrain set: {len(X_train_mj):,}")
print(f"Test set: {len(X_test_mj):,}")

# Standardize features
scaler_mj = StandardScaler()
X_train_mj_scaled = scaler_mj.fit_transform(X_train_mj)
X_test_mj_scaled = scaler_mj.transform(X_test_mj)

# Train logistic regression
lr_mj = LogisticRegression(max_iter=1000, random_state=42)
lr_mj.fit(X_train_mj_scaled, y_train_mj)

# Predictions
y_pred_mj = lr_mj.predict(X_test_mj_scaled)
y_pred_proba_mj = lr_mj.predict_proba(X_test_mj_scaled)[:, 1]

# Performance metrics
print("\n" + "-"*80)
print("MARIJUANA MODEL PERFORMANCE:")
print("-"*80)

accuracy_mj = accuracy_score(y_test_mj, y_pred_mj)
precision_mj = precision_score(y_test_mj, y_pred_mj)
recall_mj = recall_score(y_test_mj, y_pred_mj)
f1_mj = f1_score(y_test_mj, y_pred_mj)
roc_auc_mj = roc_auc_score(y_test_mj, y_pred_proba_mj)

print(f"Accuracy:  {accuracy_mj:.4f}")
print(f"Precision: {precision_mj:.4f}")
print(f"Recall:    {recall_mj:.4f}")
print(f"F1-Score:  {f1_mj:.4f}")
print(f"ROC-AUC:   {roc_auc_mj:.4f}")

# Confusion matrix
cm_mj = confusion_matrix(y_test_mj, y_pred_mj)
print(f"\nConfusion Matrix:")
print(f"                Predicted")
print(f"                No    Yes")
print(f"Actual No    {cm_mj[0,0]:5d} {cm_mj[0,1]:5d}")
print(f"Actual Yes   {cm_mj[1,0]:5d} {cm_mj[1,1]:5d}")

# Coefficients
print(f"\nModel Coefficients (Log-Odds):")
for i, feature in enumerate(features_mj):
    coef = lr_mj.coef_[0][i]
    odds_ratio = np.exp(coef)
    print(f"  {feature:20s}: β={coef:7.4f}, OR={odds_ratio:7.4f}")
print(f"  {'Intercept':20s}: β={lr_mj.intercept_[0]:7.4f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING REGRESSION VISUALIZATIONS...")
print("="*80)

import os
os.makedirs('figures', exist_ok=True)

# ============================================================================
# FIGURE 15: ROC Curves
# ============================================================================
print("\n[1/4] Creating Figure 15: ROC Curves...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Cigarette ROC
fpr_cig, tpr_cig, _ = roc_curve(y_test_cig, y_pred_proba_cig)
axes[0].plot(fpr_cig, tpr_cig, linewidth=2, label=f'ROC (AUC = {roc_auc_cig:.3f})', color=colors[0])
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curve - Cigarette Use Prediction', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Marijuana ROC
fpr_mj, tpr_mj, _ = roc_curve(y_test_mj, y_pred_proba_mj)
axes[1].plot(fpr_mj, tpr_mj, linewidth=2, label=f'ROC (AUC = {roc_auc_mj:.3f})', color=colors[2])
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].set_title('ROC Curve - Marijuana Use Prediction', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig15_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/fig15_roc_curves.png")

# ============================================================================
# FIGURE 16: Confusion Matrices
# ============================================================================
print("\n[2/4] Creating Figure 16: Confusion Matrices...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Cigarette confusion matrix
sns.heatmap(cm_cig, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
            cbar_kws={'label': 'Count'})
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('Actual', fontsize=12)
axes[0].set_title('Confusion Matrix - Cigarette Use', fontsize=14, fontweight='bold')

# Marijuana confusion matrix
sns.heatmap(cm_mj, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
            cbar_kws={'label': 'Count'})
axes[1].set_xlabel('Predicted', fontsize=12)
axes[1].set_ylabel('Actual', fontsize=12)
axes[1].set_title('Confusion Matrix - Marijuana Use', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/fig16_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/fig16_confusion_matrices.png")

# ============================================================================
# FIGURE 17: Feature Importance (Coefficients)
# ============================================================================
print("\n[3/4] Creating Figure 17: Feature Importance...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Cigarette coefficients
coef_cig = lr_cig.coef_[0]
feature_names_cig = ['Age Category', 'Gender', 'Cig Risk\nPerception', 'Mental\nDistress']
sorted_idx_cig = np.argsort(np.abs(coef_cig))[::-1]

axes[0].barh(range(len(coef_cig)), coef_cig[sorted_idx_cig], color=colors[0], alpha=0.7, edgecolor='black')
axes[0].set_yticks(range(len(coef_cig)))
axes[0].set_yticklabels([feature_names_cig[i] for i in sorted_idx_cig])
axes[0].set_xlabel('Coefficient (Log-Odds)', fontsize=12)
axes[0].set_title('Feature Importance - Cigarette Use', fontsize=14, fontweight='bold')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1)
axes[0].grid(True, alpha=0.3, axis='x')

# Marijuana coefficients
coef_mj = lr_mj.coef_[0]
feature_names_mj = ['Age Category', 'Gender', 'MJ Risk\nPerception', 'Mental\nDistress', 'Cig Ever\nUsed']
sorted_idx_mj = np.argsort(np.abs(coef_mj))[::-1]

axes[1].barh(range(len(coef_mj)), coef_mj[sorted_idx_mj], color=colors[2], alpha=0.7, edgecolor='black')
axes[1].set_yticks(range(len(coef_mj)))
axes[1].set_yticklabels([feature_names_mj[i] for i in sorted_idx_mj])
axes[1].set_xlabel('Coefficient (Log-Odds)', fontsize=12)
axes[1].set_title('Feature Importance - Marijuana Use', fontsize=14, fontweight='bold')
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1)
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('figures/fig17_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/fig17_feature_importance.png")

# ============================================================================
# FIGURE 18: Model Performance Comparison
# ============================================================================
print("\n[4/4] Creating Figure 18: Model Performance Comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
cig_scores = [accuracy_cig, precision_cig, recall_cig, f1_cig, roc_auc_cig]
mj_scores = [accuracy_mj, precision_mj, recall_mj, f1_mj, roc_auc_mj]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, cig_scores, width, label='Cigarette Model', 
               color=colors[0], alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, mj_scores, width, label='Marijuana Model', 
               color=colors[2], alpha=0.7, edgecolor='black')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figures/fig18_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/fig18_model_comparison.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("REGRESSION ANALYSIS COMPLETE!")
print("="*80)

print("\nSUMMARY OF RESULTS:")
print("-"*80)
print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
print("-"*80)
print(f"{'Cigarette Use':<20} {accuracy_cig:<12.4f} {precision_cig:<12.4f} {recall_cig:<12.4f} {f1_cig:<12.4f} {roc_auc_cig:<12.4f}")
print(f"{'Marijuana Use':<20} {accuracy_mj:<12.4f} {precision_mj:<12.4f} {recall_mj:<12.4f} {f1_mj:<12.4f} {roc_auc_mj:<12.4f}")
print("-"*80)

print("\nMOST IMPORTANT PREDICTORS:")
print("-"*80)

# Cigarette
print("\nCigarette Use (by absolute coefficient):")
for i in sorted_idx_cig[:3]:
    print(f"  {i+1}. {feature_names_cig[i]:20s}: β={coef_cig[i]:7.4f}, OR={np.exp(coef_cig[i]):7.4f}")

# Marijuana  
print("\nMarijuana Use (by absolute coefficient):")
for i in sorted_idx_mj[:3]:
    print(f"  {i+1}. {feature_names_mj[i]:20s}: β={coef_mj[i]:7.4f}, OR={np.exp(coef_mj[i]):7.4f}")

print("\n" + "="*80)
print("Generated 4 new figures:")
print("  15. fig15_roc_curves.png")
print("  16. fig16_confusion_matrices.png")
print("  17. fig17_feature_importance.png")
print("  18. fig18_model_comparison.png")
print("="*80)