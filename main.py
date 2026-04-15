# main.py
# Master script — runs the full project pipeline end to end

import subprocess
import sys

steps = [
    ("📦 Generating Dataset",     "data/generate_data.py"),
    ("🧹 Preprocessing Data",     "src/preprocess.py"),
    ("📊 Running EDA",            "src/eda.py"),
    ("🤖 Training Models",        "src/train_model.py"),
    ("🎯 Running Predictions",    "predict.py"),
]

print("=" * 55)
print("   🏢 EMPLOYEE PERFORMANCE PREDICTOR — FULL PIPELINE")
print("=" * 55)

for step_name, script in steps:
    print(f"\n▶ {step_name}...")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"❌ FAILED at: {script}")
        print("Fix the error above, then re-run main.py.")
        sys.exit(1)
    print(f"✅ {step_name} — Done!")

print("\n" + "=" * 55)
print("   🎉 FULL PIPELINE COMPLETE!")
print("   Run: streamlit run app.py")
print("=" * 55)