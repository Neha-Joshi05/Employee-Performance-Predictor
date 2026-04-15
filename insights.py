# ============================================================
# insights.py - HR Insights & Business Recommendations
# Phase 8: Turn predictions into actionable HR decisions
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "outputs"
IMAGE_DIR  = "images"
MODEL_DIR  = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR,  exist_ok=True)

print("=" * 55)
print("   📊 HR INSIGHTS & BUSINESS RECOMMENDATIONS")
print("=" * 55)

# ── Load data ─────────────────────────────────────────────────
pred_path = os.path.join(OUTPUT_DIR, "predictions.csv")
if not os.path.exists(pred_path):
    raise FileNotFoundError("Run predict.py first to generate predictions.csv")

df = pd.read_csv(pred_path)
print(f"✅ Loaded {len(df)} employee records with predictions.\n")

perf_col = "predicted_performance"


# ══════════════════════════════════════════════════════════════
# INSIGHT 1 — Performance Distribution
# ══════════════════════════════════════════════════════════════
dist = df[perf_col].value_counts(normalize=True) * 100
print("─" * 55)
print("📌 INSIGHT 1: Performance Distribution")
print("─" * 55)
for band, pct in dist.items():
    bar = "█" * int(pct / 2)
    print(f"  {band:8s} : {bar} {pct:.1f}%")


# ══════════════════════════════════════════════════════════════
# INSIGHT 2 — Department-wise Performance
# ══════════════════════════════════════════════════════════════
print("\n─" * 28)
print("📌 INSIGHT 2: Department-wise High Performers")
print("─" * 55)

if 'department' in df.columns:
    dept_high = (
        df[df[perf_col] == "High"]
        .groupby("department")
        .size()
        .reset_index(name="high_performers")
    )
    dept_total = df.groupby("department").size().reset_index(name="total")
    dept_df = dept_high.merge(dept_total, on="department")
    dept_df["high_%"] = (dept_df["high_performers"] / dept_df["total"] * 100).round(1)
    dept_df = dept_df.sort_values("high_%", ascending=False)

    for _, row in dept_df.iterrows():
        print(f"  {row['department']:15s} → {row['high_%']:5.1f}% High  ({row['high_performers']}/{row['total']})")

    top_dept    = dept_df.iloc[0]["department"]
    bottom_dept = dept_df.iloc[-1]["department"]
    print(f"\n  🏆 Best dept  : {top_dept}")
    print(f"  ⚠️  Needs focus: {bottom_dept}")


# ══════════════════════════════════════════════════════════════
# INSIGHT 3 — Salary vs Performance
# ══════════════════════════════════════════════════════════════
print("\n─" * 28)
print("📌 INSIGHT 3: Salary vs Performance Band")
print("─" * 55)

if 'salary' in df.columns:
    sal = df.groupby(perf_col)["salary"].mean().reindex(["Low","Medium","High"])
    for band, avg in sal.items():
        print(f"  {band:8s} avg salary: ₹{avg:,.0f}")
    print("\n  💡 Insight: Low performers may be overpaid for output — review compensation strategy.")


# ══════════════════════════════════════════════════════════════
# INSIGHT 4 — Training Hours Impact
# ══════════════════════════════════════════════════════════════
print("\n─" * 28)
print("📌 INSIGHT 4: Training Hours vs Performance")
print("─" * 55)

if 'training_hours' in df.columns:
    trn = df.groupby(perf_col)["training_hours"].mean().reindex(["Low","Medium","High"])
    for band, avg in trn.items():
        print(f"  {band:8s} avg training: {avg:.1f} hrs/yr")
    print("\n  💡 Insight: High performers log more training hours. Invest in L&D programs for Low/Medium bands.")


# ══════════════════════════════════════════════════════════════
# INSIGHT 5 — Satisfaction & Retention Risk
# ══════════════════════════════════════════════════════════════
print("\n─" * 28)
print("📌 INSIGHT 5: Satisfaction Score vs Performance")
print("─" * 55)

if 'satisfaction_score' in df.columns:
    sat = df.groupby(perf_col)["satisfaction_score"].mean().reindex(["Low","Medium","High"])
    for band, avg in sat.items():
        print(f"  {band:8s} avg satisfaction: {avg:.2f}/10")

    low_sat_high = df[(df[perf_col] == "High") & (df["satisfaction_score"] < 5)]
    print(f"\n  ⚠️  High performers with LOW satisfaction: {len(low_sat_high)} employees")
    print(  "     → RETENTION RISK! Prioritise engagement for these employees.")


# ══════════════════════════════════════════════════════════════
# INSIGHT 6 — Absenteeism Risk
# ══════════════════════════════════════════════════════════════
print("\n─" * 28)
print("📌 INSIGHT 6: Absenteeism Analysis")
print("─" * 55)

if 'absenteeism_days' in df.columns:
    ab = df.groupby(perf_col)["absenteeism_days"].mean().reindex(["Low","Medium","High"])
    for band, avg in ab.items():
        print(f"  {band:8s} avg absent days: {avg:.1f}")
    high_ab = df[df["absenteeism_days"] > 10]
    print(f"\n  ⚠️  Employees with >10 absent days: {len(high_ab)} ({len(high_ab)/len(df)*100:.1f}%)")
    print(  "     → Flag for HR wellbeing check-in.")


# ══════════════════════════════════════════════════════════════
# INSIGHT 7 — Personalised Action Plan per Employee
# ══════════════════════════════════════════════════════════════
print("\n─" * 28)
print("📌 INSIGHT 7: Personalised HR Action Plans (sample 5)")
print("─" * 55)

def action_plan(row):
    actions = []
    band = row.get(perf_col, "Medium")

    if band == "Low":
        actions.append("📚 Enrol in mandatory upskilling program (30+ hrs/yr)")
        actions.append("🤝 Assign performance improvement plan (PIP) with 90-day review")
        if row.get("absenteeism_days", 0) > 8:
            actions.append("❤️  Refer to Employee Assistance Programme (EAP)")
        if row.get("satisfaction_score", 5) < 5:
            actions.append("💬 1-on-1 manager counselling session — understand root cause")

    elif band == "Medium":
        actions.append("🎯 Set stretch goals for next quarter")
        actions.append("📊 Monthly progress check-in with manager")
        if row.get("training_hours", 0) < 20:
            actions.append("🖥️  Assign relevant online certification (Coursera / Udemy)")

    elif band == "High":
        actions.append("🌟 Fast-track promotion consideration")
        actions.append("🏆 Recognise in team meeting / rewards programme")
        if row.get("satisfaction_score", 5) < 6:
            actions.append("⚠️  Retention risk — discuss career path & compensation review")
        actions.append("👨‍🏫 Consider for mentorship / team-lead role")

    return " | ".join(actions)

df["hr_action_plan"] = df.apply(action_plan, axis=1)

# Print 5 sample action plans
sample = df.sample(5, random_state=42) if len(df) >= 5 else df
for _, row in sample.iterrows():
    name = row.get("employee_id", f"EMP-{_}")
    print(f"\n  [{row[perf_col]}] {name}")
    for a in row["hr_action_plan"].split(" | "):
        print(f"    → {a}")


# ══════════════════════════════════════════════════════════════
# VISUALIZATION — Insights Dashboard (4 plots)
# ══════════════════════════════════════════════════════════════
print("\n\n🎨 Generating Insights Dashboard...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("HR Insights Dashboard — Employee Performance Predictor",
             fontsize=15, fontweight='bold', y=1.01)

colors = {"High": "#2ecc71", "Medium": "#f39c12", "Low": "#e74c3c"}
order  = ["Low", "Medium", "High"]

# Plot 1 — Performance Distribution (pie)
ax = axes[0, 0]
sizes = [dist.get(b, 0) for b in order]
clrs  = [colors[b] for b in order]
ax.pie(sizes, labels=order, colors=clrs, autopct='%1.1f%%',
       startangle=140, textprops={'fontsize': 11})
ax.set_title("Performance Band Distribution", fontweight='bold')

# Plot 2 — Dept High-Performer %
ax = axes[0, 1]
if 'department' in df.columns:
    sns.barplot(data=dept_df, x="high_%", y="department",
                palette="Blues_d", ax=ax)
    ax.set_title("% High Performers by Department", fontweight='bold')
    ax.set_xlabel("High Performer %")
    ax.set_ylabel("")
    for p in ax.patches:
        ax.text(p.get_width() + 0.3, p.get_y() + p.get_height()/2,
                f"{p.get_width():.1f}%", va='center', fontsize=9)

# Plot 3 — Training Hours by Performance
ax = axes[1, 0]
if 'training_hours' in df.columns:
    sns.boxplot(data=df, x=perf_col, y='training_hours',
                order=order, palette=colors, ax=ax)
    ax.set_title("Training Hours vs Performance Band", fontweight='bold')
    ax.set_xlabel("Performance Band")
    ax.set_ylabel("Training Hours / Year")

# Plot 4 — Satisfaction Score by Performance
ax = axes[1, 1]
if 'satisfaction_score' in df.columns:
    sns.violinplot(data=df, x=perf_col, y='satisfaction_score',
                   order=order, palette=colors, inner='box', ax=ax)
    ax.set_title("Satisfaction Score vs Performance Band", fontweight='bold')
    ax.set_xlabel("Performance Band")
    ax.set_ylabel("Satisfaction Score (1–10)")

plt.tight_layout()
img_path = os.path.join(IMAGE_DIR, "12_hr_insights_dashboard.png")
plt.savefig(img_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Dashboard saved → {img_path}")


# ══════════════════════════════════════════════════════════════
# SAVE full report with action plans
# ══════════════════════════════════════════════════════════════
report_path = os.path.join(OUTPUT_DIR, "hr_insights_report.csv")
df.to_csv(report_path, index=False)
print(f"✅ Full HR report saved → {report_path}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  📋 EXECUTIVE SUMMARY")
print("=" * 55)
total      = len(df)
high_count = (df[perf_col] == "High").sum()
low_count  = (df[perf_col] == "Low").sum()
risk_count = len(df[(df[perf_col] == "High") & (df.get("satisfaction_score", pd.Series([5]*total)) < 5)])

print(f"  Total Employees Analysed : {total}")
print(f"  High Performers          : {high_count} ({high_count/total*100:.1f}%)")
print(f"  Low Performers (need PIP): {low_count}  ({low_count/total*100:.1f}%)")
print(f"  Retention Risks          : {risk_count} high performers with low satisfaction")
print(f"  Top Department           : {top_dept if 'department' in df.columns else 'N/A'}")
print("=" * 55)
print("\n✅ Phase 8 Complete — HR Insights Ready!\n")