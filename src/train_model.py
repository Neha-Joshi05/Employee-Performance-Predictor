# src/train_model.py
# Feature Engineering + Model Training + Evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import joblib
import os

os.makedirs('models', exist_ok=True)
os.makedirs('images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# -------------------------------------------------------
# STEP 1: Load Preprocessed Data
# -------------------------------------------------------
X = pd.read_csv('outputs/X_scaled.csv')
y = pd.read_csv('outputs/y_target.csv').squeeze()

print("✅ Data loaded")
print(f"Features : {X.shape[1]} | Samples : {X.shape[0]}")
print(f"Target Classes : {sorted(y.unique())}  (0=High, 1=Low, 2=Medium)\n")

# -------------------------------------------------------
# STEP 2: Feature Engineering (add 3 new smart features)
# -------------------------------------------------------
# Load original cleaned data for engineering
df_raw = pd.read_csv('outputs/cleaned_data.csv')

# Feature 1: productivity_index — projects per year at company
df_raw['productivity_index'] = (
    df_raw['projects_completed'] / (df_raw['years_at_company'] + 1)
).round(3)

# Feature 2: engagement_score — combo of satisfaction + manager rating
df_raw['engagement_score'] = (
    (df_raw['satisfaction_score'] + df_raw['manager_rating']) / 2
).round(3)

# Feature 3: overwork_flag — 1 if overtime > 30 hrs AND satisfaction < 3
df_raw['overwork_flag'] = (
    (df_raw['overtime_hours'] > 30) & (df_raw['satisfaction_score'] < 3)
).astype(int)

new_features = ['productivity_index', 'engagement_score', 'overwork_flag']
print(f"✅ Feature Engineering done | New features added: {new_features}")

# Scale new features using saved scaler (fit fresh on new cols only)
from sklearn.preprocessing import StandardScaler
scaler_new = StandardScaler()
new_scaled = scaler_new.fit_transform(df_raw[new_features])
new_df = pd.DataFrame(new_scaled, columns=new_features)

# Combine with existing scaled features
X_final = pd.concat([X.reset_index(drop=True), new_df], axis=1)
print(f"✅ Final feature count: {X_final.shape[1]}\n")

# Save engineered feature names
joblib.dump(list(X_final.columns), 'models/feature_names.pkl')

# -------------------------------------------------------
# STEP 3: Train-Test Split
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}\n")

# -------------------------------------------------------
# STEP 4: Train Model 1 — Logistic Regression
# -------------------------------------------------------
print("🔄 Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
lr_cv = cross_val_score(lr, X_final, y, cv=5).mean()

print(f"✅ Logistic Regression")
print(f"   Test Accuracy  : {lr_acc*100:.2f}%")
print(f"   CV Accuracy    : {lr_cv*100:.2f}%")
print(f"   Report:\n{classification_report(y_test, lr_pred, target_names=['High','Low','Medium'])}")

# -------------------------------------------------------
# STEP 5: Train Model 2 — Random Forest
# -------------------------------------------------------
print("🔄 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_cv = cross_val_score(rf, X_final, y, cv=5).mean()

print(f"✅ Random Forest")
print(f"   Test Accuracy  : {rf_acc*100:.2f}%")
print(f"   CV Accuracy    : {rf_cv*100:.2f}%")
print(f"   Report:\n{classification_report(y_test, rf_pred, target_names=['High','Low','Medium'])}")

# -------------------------------------------------------
# STEP 6: Save Best Model
# -------------------------------------------------------
best_model = rf if rf_acc >= lr_acc else lr
best_name = "Random Forest" if rf_acc >= lr_acc else "Logistic Regression"
joblib.dump(best_model, 'models/best_model.pkl')
print(f"✅ Best Model: {best_name} saved to models/best_model.pkl\n")

# -------------------------------------------------------
# STEP 7: Confusion Matrix Plot (Random Forest)
# -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, model, preds, name in zip(
    axes,
    [lr, rf],
    [lr_pred, rf_pred],
    ['Logistic Regression', 'Random Forest']
):
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['High', 'Low', 'Medium'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{name}\nAccuracy: {accuracy_score(y_test, preds)*100:.2f}%',
                 fontsize=12, fontweight='bold')

plt.suptitle('Confusion Matrix Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/09_confusion_matrices.png', dpi=150)
plt.close()
print("✅ Plot 9 saved: Confusion Matrices")

# -------------------------------------------------------
# STEP 8: Feature Importance Plot (Random Forest)
# -------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=X_final.columns)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(10, 8))
colors = ['#e74c3c' if i >= len(importances)-3 else '#3498db'
          for i in range(len(importances))]
importances.plot(kind='barh', color=colors, edgecolor='black')
plt.title('Feature Importance — Random Forest\n(Red = Top 3 Most Important)',
          fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('images/10_feature_importance.png', dpi=150)
plt.close()
print("✅ Plot 10 saved: Feature Importance")

# -------------------------------------------------------
# STEP 9: Model Comparison Bar Chart
# -------------------------------------------------------
plt.figure(figsize=(7, 5))
models = ['Logistic Regression', 'Random Forest']
accs = [lr_acc * 100, rf_acc * 100]
bars = plt.bar(models, accs, color=['#3498db', '#2ecc71'], edgecolor='black', width=0.4)
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{acc:.2f}%', ha='center', fontsize=12, fontweight='bold')
plt.ylim(0, 115)
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.tight_layout()
plt.savefig('images/11_model_comparison.png', dpi=150)
plt.close()
print("✅ Plot 11 saved: Model Comparison")

# -------------------------------------------------------
# FINAL SUMMARY
# -------------------------------------------------------
print("\n" + "="*50)
print("        🏆 MODEL TRAINING SUMMARY")
print("="*50)
print(f"  Logistic Regression  : {lr_acc*100:.2f}% (CV: {lr_cv*100:.2f}%)")
print(f"  Random Forest        : {rf_acc*100:.2f}% (CV: {rf_cv*100:.2f}%)")
print(f"  Best Model Saved     : {best_name}")
print(f"  Total Features Used  : {X_final.shape[1]}")
print("="*50)
print("\n✅ Phase 5 & 6 Complete!")