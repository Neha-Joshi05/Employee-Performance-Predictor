# src/eda.py
# Exploratory Data Analysis - generates insights and visualizations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('images', exist_ok=True)

# --- Load cleaned data ---
df = pd.read_csv('outputs/cleaned_data.csv')
print("✅ Data loaded for EDA")
print(f"Shape: {df.shape}\n")

# Plot style
sns.set_theme(style="whitegrid")
PALETTE = {'High': '#2ecc71', 'Medium': '#f39c12', 'Low': '#e74c3c'}

# -------------------------------------------------------
# PLOT 1: Performance Label Distribution (Bar Chart)
# -------------------------------------------------------
plt.figure(figsize=(7, 5))
order = ['High', 'Medium', 'Low']
counts = df['performance_label'].value_counts().reindex(order)
bars = plt.bar(counts.index, counts.values,
               color=[PALETTE[k] for k in order], edgecolor='black', width=0.5)
for bar, val in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(val), ha='center', fontweight='bold', fontsize=12)
plt.title('Employee Performance Distribution', fontsize=15, fontweight='bold')
plt.xlabel('Performance Level')
plt.ylabel('Number of Employees')
plt.tight_layout()
plt.savefig('images/01_performance_distribution.png', dpi=150)
plt.close()
print("✅ Plot 1 saved: Performance Distribution")

# -------------------------------------------------------
# PLOT 2: Age Distribution by Performance
# -------------------------------------------------------
plt.figure(figsize=(9, 5))
for label in order:
    subset = df[df['performance_label'] == label]['age']
    plt.hist(subset, bins=15, alpha=0.6, label=label, color=PALETTE[label], edgecolor='black')
plt.title('Age Distribution by Performance Level', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('images/02_age_distribution.png', dpi=150)
plt.close()
print("✅ Plot 2 saved: Age Distribution")

# -------------------------------------------------------
# PLOT 3: Department vs Performance (Stacked Bar)
# -------------------------------------------------------
dept_perf = df.groupby(['department', 'performance_label']).size().unstack(fill_value=0)
dept_perf = dept_perf[order]
dept_perf.plot(kind='bar', stacked=True, figsize=(10, 6),
               color=[PALETTE[k] for k in order], edgecolor='black')
plt.title('Department vs Performance Level', fontsize=14, fontweight='bold')
plt.xlabel('Department')
plt.ylabel('Number of Employees')
plt.xticks(rotation=30)
plt.legend(title='Performance')
plt.tight_layout()
plt.savefig('images/03_department_performance.png', dpi=150)
plt.close()
print("✅ Plot 3 saved: Department vs Performance")

# -------------------------------------------------------
# PLOT 4: Training Hours vs Performance (Boxplot)
# -------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='performance_label', y='training_hours',
            order=order, palette=PALETTE)
plt.title('Training Hours by Performance Level', fontsize=14, fontweight='bold')
plt.xlabel('Performance Level')
plt.ylabel('Training Hours')
plt.tight_layout()
plt.savefig('images/04_training_hours_boxplot.png', dpi=150)
plt.close()
print("✅ Plot 4 saved: Training Hours Boxplot")

# -------------------------------------------------------
# PLOT 5: Salary vs Performance (Boxplot)
# -------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='performance_label', y='monthly_salary',
            order=order, palette=PALETTE)
plt.title('Monthly Salary by Performance Level', fontsize=14, fontweight='bold')
plt.xlabel('Performance Level')
plt.ylabel('Monthly Salary (₹)')
plt.tight_layout()
plt.savefig('images/05_salary_boxplot.png', dpi=150)
plt.close()
print("✅ Plot 5 saved: Salary Boxplot")

# -------------------------------------------------------
# PLOT 6: Correlation Heatmap
# -------------------------------------------------------
num_cols = ['age', 'years_at_company', 'monthly_salary', 'training_hours',
            'projects_completed', 'overtime_hours', 'satisfaction_score',
            'manager_rating', 'attendance_rate', 'last_promotion_years',
            'performance_score']

plt.figure(figsize=(12, 9))
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            linewidths=0.5, annot_kws={"size": 9})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/06_correlation_heatmap.png', dpi=150)
plt.close()
print("✅ Plot 6 saved: Correlation Heatmap")

# -------------------------------------------------------
# PLOT 7: Manager Rating vs Performance (Violin Plot)
# -------------------------------------------------------
plt.figure(figsize=(9, 5))
sns.violinplot(data=df, x='performance_label', y='manager_rating',
               order=order, palette=PALETTE, inner='quartile')
plt.title('Manager Rating by Performance Level', fontsize=14, fontweight='bold')
plt.xlabel('Performance Level')
plt.ylabel('Manager Rating')
plt.tight_layout()
plt.savefig('images/07_manager_rating_violin.png', dpi=150)
plt.close()
print("✅ Plot 7 saved: Manager Rating Violin")

# -------------------------------------------------------
# PLOT 8: Satisfaction Score vs Performance Score (Scatter)
# -------------------------------------------------------
plt.figure(figsize=(9, 6))
for label in order:
    subset = df[df['performance_label'] == label]
    plt.scatter(subset['satisfaction_score'], subset['performance_score'],
                alpha=0.5, label=label, color=PALETTE[label], s=30)
plt.title('Satisfaction Score vs Performance Score', fontsize=14, fontweight='bold')
plt.xlabel('Satisfaction Score')
plt.ylabel('Performance Score')
plt.legend(title='Performance')
plt.tight_layout()
plt.savefig('images/08_satisfaction_vs_performance.png', dpi=150)
plt.close()
print("✅ Plot 8 saved: Satisfaction vs Performance Scatter")

# -------------------------------------------------------
# KEY INSIGHTS SUMMARY
# -------------------------------------------------------
print("\n" + "="*55)
print("         📊 KEY EDA INSIGHTS SUMMARY")
print("="*55)
print(f"Total Employees Analyzed : {len(df)}")
print(f"High Performers          : {(df['performance_label']=='High').sum()} ({(df['performance_label']=='High').mean()*100:.1f}%)")
print(f"Medium Performers        : {(df['performance_label']=='Medium').sum()} ({(df['performance_label']=='Medium').mean()*100:.1f}%)")
print(f"Low Performers           : {(df['performance_label']=='Low').sum()} ({(df['performance_label']=='Low').mean()*100:.1f}%)")
print("-"*55)

for label in order:
    subset = df[df['performance_label'] == label]
    print(f"\n[{label} Performers]")
    print(f"  Avg Manager Rating   : {subset['manager_rating'].mean():.2f}")
    print(f"  Avg Training Hours   : {subset['training_hours'].mean():.1f}")
    print(f"  Avg Salary           : ₹{subset['monthly_salary'].mean():,.0f}")
    print(f"  Avg Satisfaction     : {subset['satisfaction_score'].mean():.2f}")
    print(f"  Avg Projects Done    : {subset['projects_completed'].mean():.1f}")

print("\n✅ EDA Complete! All 8 charts saved to images/")