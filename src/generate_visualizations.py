"""
Script to train models, evaluate performance metrics, and generate
high-resolution publication-quality visualization charts for GitHub README.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

ASSETS_DIR = "assets"
MODELS_DIR = "models"
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Aesthetic styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.dpi'] = 300

# 1. Load data
DATA_PATH = "data/loan_data.csv"
df = pd.read_csv(DATA_PATH)
df_clean = df.drop(columns=['Loan_ID']).copy()

raw_status_counts = df['Loan_Status'].value_counts()

# ----------------- Preprocessing -----------------
# Imputation
cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
for col in cat_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

num_cols = ['LoanAmount', 'Loan_Amount_Term']
for col in num_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

df_clean['Dependents'] = df_clean['Dependents'].replace('3+', '3')

categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])

X = df_clean.drop(columns=['Loan_Status'])
y = df_clean['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------- Model Training & Caching -----------------
# 1. Logistic Regression
lr_path = os.path.join(MODELS_DIR, "logistic_regression.pkl")
if os.path.exists(lr_path):
    with open(lr_path, "rb") as f:
        lr_model = pickle.load(f)
else:
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    with open(lr_path, "wb") as f:
        pickle.dump(lr_model, f)

lr_pred = lr_model.predict(X_test)
lr_scores = lr_model.predict_proba(X_test)[:, 1]

# 2. Random Forest
rf_path = os.path.join(MODELS_DIR, "loan_approval_model.pkl")
if os.path.exists(rf_path):
    with open(rf_path, "rb") as f:
        rf_model = pickle.load(f)
else:
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    with open(rf_path, "wb") as f:
        pickle.dump(rf_model, f)

rf_pred = rf_model.predict(X_test)
rf_scores = rf_model.predict_proba(X_test)[:, 1]

# 3. SVM
svm_path = os.path.join(MODELS_DIR, "svm_model.pkl")
if os.path.exists(svm_path):
    print("Loading cached SVM model...")
    with open(svm_path, "rb") as f:
        svm_model = pickle.load(f)
else:
    print("Fitting SVM model (may take ~1.5 - 2 minutes)...")
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    with open(svm_path, "wb") as f:
        pickle.dump(svm_model, f)
    print("SVM model cached successfully.")

svm_pred = svm_model.predict(X_test)
svm_scores = svm_model.decision_function(X_test)

models = {
    "Logistic Regression": (lr_model, lr_pred, lr_scores),
    "Random Forest": (rf_model, rf_pred, rf_scores),
    "Support Vector Machine": (svm_model, svm_pred, svm_scores),
}

# ----------------- Metrics Computation -----------------
metrics_summary = []
for name, (mod, pred, scores) in models.items():
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)
    metrics_summary.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })

metrics_df = pd.DataFrame(metrics_summary)
print("\n=== Model Metrics Summary ===")
print(metrics_df.to_string(index=False))

# ----------------- Visualizations -----------------

# 1. Target Class Distribution
fig, ax = plt.subplots(figsize=(7, 5), facecolor='#ffffff')
colors = ['#10b981', '#ef4444']
labels = ['Approved (Y)', 'Rejected (N)']
sizes = [raw_status_counts['Y'], raw_status_counts['N']]
explode = (0.04, 0.04)

wedges, texts, autotexts = ax.pie(
    sizes,
    explode=explode,
    labels=labels,
    autopct='%1.1f%%',
    pctdistance=0.75,
    startangle=140,
    colors=colors,
    textprops=dict(color="#1f2937", fontweight='bold', fontsize=11),
    wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
)

for autotext in autotexts:
    autotext.set_fontsize(13)
    autotext.set_weight('bold')
    autotext.set_color('#ffffff')

ax.set_title("Loan Application Approval Distribution\n(Total Records: 614)", pad=20, color='#111827')
ax.legend(
    wedges,
    [f"Approved: {sizes[0]} ({sizes[0]/sum(sizes)*100:.1f}%)",
     f"Rejected: {sizes[1]} ({sizes[1]/sum(sizes)*100:.1f}%)"],
    title="Loan Status",
    loc="lower center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    frameon=True,
    facecolor='#f8fafc',
    edgecolor='#e2e8f0'
)
plt.tight_layout()
plt.savefig(f"{ASSETS_DIR}/loan_status_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: loan_status_distribution.png")

# 2. Credit History Impact on Approval
fig, ax = plt.subplots(figsize=(8, 5), facecolor='#ffffff')
credit_analysis = df.dropna(subset=['Credit_History']).copy()
credit_analysis['Credit_History_Label'] = credit_analysis['Credit_History'].map({1.0: 'Has Credit History (1.0)', 0.0: 'No Credit History (0.0)'})
credit_cross = pd.crosstab(credit_analysis['Credit_History_Label'], credit_analysis['Loan_Status'], normalize='index') * 100

credit_cross[['Y', 'N']].plot(
    kind='bar',
    stacked=False,
    color=['#10b981', '#ef4444'],
    ax=ax,
    width=0.6,
    edgecolor='none'
)
ax.set_title("Impact of Credit History on Loan Approval Rate", pad=15, color='#111827')
ax.set_xlabel("Applicant Credit History Status", labelpad=10)
ax.set_ylabel("Approval Percentage (%)", labelpad=10)
ax.set_ylim(0, 100)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(['Approved (%)', 'Rejected (%)'], frameon=True, facecolor='#f8fafc', edgecolor='#e2e8f0')
ax.grid(axis='y', linestyle='--', alpha=0.5)

for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', padding=4, fontweight='bold', fontsize=10, color='#374151')

plt.tight_layout()
plt.savefig(f"{ASSETS_DIR}/credit_history_impact.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: credit_history_impact.png")

# 3. Applicant Income vs Loan Amount by Status
fig, ax = plt.subplots(figsize=(8, 5.5), facecolor='#ffffff')
scatter_df = df[(df['ApplicantIncome'] < 25000) & (df['LoanAmount'] < 500)].copy()

sns.scatterplot(
    data=scatter_df,
    x='ApplicantIncome',
    y='LoanAmount',
    hue='Loan_Status',
    palette={'Y': '#10b981', 'N': '#ef4444'},
    alpha=0.75,
    s=55,
    edgecolor='white',
    linewidth=0.5,
    ax=ax
)
ax.set_title("Applicant Income vs. Loan Amount Requested", pad=15, color='#111827')
ax.set_xlabel("Applicant Income ($)", labelpad=10)
ax.set_ylabel("Loan Amount (Thousands $)", labelpad=10)
ax.legend(title="Loan Status", labels=['Approved (Y)', 'Rejected (N)'], frameon=True, facecolor='#f8fafc', edgecolor='#e2e8f0')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{ASSETS_DIR}/income_vs_loan.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: income_vs_loan.png")

# 4. Feature Importance (Random Forest)
fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='#ffffff')
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(importances)))
bars = ax.barh(importances['Feature'], importances['Importance'], color=colors, height=0.65)
ax.set_title("Feature Importance Ranking (Random Forest Classifier)", pad=15, color='#111827')
ax.set_xlabel("Relative Importance Weight", labelpad=10)
ax.set_ylabel("Features", labelpad=10)
ax.grid(axis='x', linestyle='--', alpha=0.5)

for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{width * 100:.1f}%",
        va='center',
        ha='left',
        fontsize=10,
        fontweight='bold',
        color='#1f2937'
    )

ax.set_xlim(0, max(importances['Importance']) * 1.15)
plt.tight_layout()
plt.savefig(f"{ASSETS_DIR}/feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: feature_importance.png")

# 5. Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#ffffff')
plot_metrics = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

palette = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#06b6d4"]
sns.barplot(
    data=plot_metrics,
    x="Metric",
    y="Score",
    hue="Model",
    palette=palette,
    ax=ax,
    edgecolor="none"
)

ax.set_title("Model Performance Benchmark Comparison", pad=15, color='#111827')
ax.set_xlabel("Evaluation Metric", labelpad=10)
ax.set_ylabel("Score (0.0 - 1.0)", labelpad=10)
ax.set_ylim(0, 1.15)
ax.legend(title="Algorithm", loc="lower right", frameon=True, facecolor='#f8fafc', edgecolor='#e2e8f0')
ax.grid(axis='y', linestyle='--', alpha=0.5)

for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9, fontweight='bold', color='#374151')

plt.tight_layout()
plt.savefig(f"{ASSETS_DIR}/model_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: model_comparison.png")

# 6. Confusion Matrices (Random Forest & Logistic Regression side-by-side)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#ffffff')

cm_rf = confusion_matrix(y_test, rf_pred)
cm_lr = confusion_matrix(y_test, lr_pred)

# Random Forest CM
sns.heatmap(
    cm_rf,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    ax=axes[0],
    xticklabels=['Rejected (0)', 'Approved (1)'],
    yticklabels=['Rejected (0)', 'Approved (1)'],
    annot_kws={"size": 14, "weight": "bold"}
)
axes[0].set_title("Confusion Matrix: Random Forest (Acc: 82.93%)", pad=12, color='#111827')
axes[0].set_xlabel("Predicted Class", labelpad=8)
axes[0].set_ylabel("Actual Class", labelpad=8)

# Logistic Regression CM
sns.heatmap(
    cm_lr,
    annot=True,
    fmt='d',
    cmap='Greens',
    cbar=False,
    ax=axes[1],
    xticklabels=['Rejected (0)', 'Approved (1)'],
    yticklabels=['Rejected (0)', 'Approved (1)'],
    annot_kws={"size": 14, "weight": "bold"}
)
axes[1].set_title("Confusion Matrix: Logistic Regression (Acc: 86.18%)", pad=12, color='#111827')
axes[1].set_xlabel("Predicted Class", labelpad=8)
axes[1].set_ylabel("Actual Class", labelpad=8)

plt.tight_layout()
plt.savefig(f"{ASSETS_DIR}/confusion_matrices.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrices.png")

# 7. ROC Curves
fig, ax = plt.subplots(figsize=(8, 6), facecolor='#ffffff')
line_colors = {'Logistic Regression': '#2563eb', 'Random Forest': '#10b981', 'Support Vector Machine': '#f59e0b'}

for name, (mod, pred, scores) in models.items():
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=line_colors[name], lw=2.5)

ax.plot([0, 1], [0, 1], color='#9ca3af', linestyle='--', lw=1.5, label='Random Chance (AUC = 0.500)')
ax.set_title("Receiver Operating Characteristic (ROC) Curves", pad=15, color='#111827')
ax.set_xlabel("False Positive Rate (1 - Specificity)", labelpad=10)
ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", labelpad=10)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
ax.legend(loc="lower right", frameon=True, facecolor='#f8fafc', edgecolor='#e2e8f0', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f"{ASSETS_DIR}/roc_curves.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: roc_curves.png")

# 8. Correlation Heatmap
fig, ax = plt.subplots(figsize=(9, 7), facecolor='#ffffff')
corr = df_clean.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-0.4,
    vmax=1.0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    ax=ax
)
ax.set_title("Correlation Heatmap of Preprocessed Features", pad=15, color='#111827')
plt.tight_layout()
plt.savefig(f"{ASSETS_DIR}/correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: correlation_matrix.png")

print("\nAll visualizations generated successfully in ./assets/!")
