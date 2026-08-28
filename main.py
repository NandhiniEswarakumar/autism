"""
Main pipeline orchestrator for the project.

This script runs the data-processing, training, evaluation and explainability
steps in sequence. Use `--step` to run specific steps, `--skip` to omit steps,
and `--app` to launch the Streamlit app after the pipeline.

Usage examples:
    python main.py
    python main.py --step 5
    python main.py --step 5 6 7
    python main.py --skip 7 8
"""

import os
import sys
import time
import argparse
import subprocess
import json

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
PYTHON = sys.executable
ROOT   = os.path.dirname(os.path.abspath(__file__))
SRC    = os.path.join(ROOT, "src")

STEPS = {
    1  : ("EDA",                  os.path.join(SRC, "01_eda.py")),
    2  : ("Preprocessing",        os.path.join(SRC, "02_preprocessing.py")),
    3  : ("Feature Selection",    os.path.join(SRC, "03_feature_selection.py")),
    4  : ("TabM Architecture",    os.path.join(SRC, "04_tabm_model.py")),
    5  : ("Training + HPO",       os.path.join(SRC, "05_train.py")),
    6  : ("Evaluation",           os.path.join(SRC, "06_evaluate.py")),
    7  : ("SHAP Explainability",  os.path.join(SRC, "07_shap_explain.py")),
    8  : ("Baseline Comparison",  os.path.join(SRC, "08_compare_models.py")),
    9  : ("Ablation Study",       os.path.join(SRC, "09_ablation.py")),
    10 : ("Robustness Testing",   os.path.join(SRC, "10_robustness.py")),
}

# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="ASD Detection — Full Pipeline",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python main.py               # Run all steps
  python main.py --step 5      # Run only training
  python main.py --step 5 6 7  # Run steps 5, 6, 7
  python main.py --skip 7 8    # Skip SHAP and comparison
  python main.py --list        # Show all steps
"""
)
parser.add_argument("--step",  type=int, nargs="+",
                    help="Run only these step numbers")
parser.add_argument("--skip",  type=int, nargs="+",
                    help="Skip these step numbers")
parser.add_argument("--list",  action="store_true",
                    help="List all pipeline steps and exit")
parser.add_argument("--app",   action="store_true",
                    help="Launch Streamlit app after pipeline")
args = parser.parse_args()

# -------------------------------------------------------------
# HEADER
# -------------------------------------------------------------
def header(text, char="=", width=62):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")

header("ASD DETECTION — FULL PIPELINE", "=")
print(f"  Python   : {PYTHON}")
print(f"  Root     : {ROOT}")
print(f"  Source   : {SRC}")

# -------------------------------------------------------------
# LIST MODE
# -------------------------------------------------------------
if args.list:
    print("\n  Pipeline Steps:")
    for num, (name, path) in STEPS.items():
        exists = "[OK]" if os.path.exists(path) else "[X]"
        print(f"  [{exists}] Step {num:>2} — {name:<25} -> {os.path.basename(path)}")
    sys.exit(0)

# -------------------------------------------------------------
# DETERMINE WHICH STEPS TO RUN
# -------------------------------------------------------------
to_run = set(STEPS.keys())
if args.step:
    to_run = {s for s in args.step if s in STEPS}
if args.skip:
    to_run -= set(args.skip)

steps_ordered = sorted(to_run)
print(f"\n  Steps to run : {steps_ordered}")

# -------------------------------------------------------------
# RUN STEPS
# -------------------------------------------------------------
results_log = {}

for step_num in steps_ordered:
    name, script = STEPS[step_num]

    if not os.path.exists(script):
        print(f"\n  ⚠️  Step {step_num} script not found: {script}")
        results_log[step_num] = {"status": "NOT_FOUND", "time_s": 0}
        continue

    header(f"STEP {step_num} — {name.upper()}", "-")
    t_start = time.time()

    try:
        result = subprocess.run(
            [PYTHON, script],
            cwd=ROOT,
            text=True,
            capture_output=False,   # print live output
        )
        elapsed = time.time() - t_start

        if result.returncode == 0:
            status = "[OK]  SUCCESS"
            results_log[step_num] = {"status": "SUCCESS", "time_s": round(elapsed, 1)}
        else:
            status = "[X]  FAILED"
            results_log[step_num] = {"status": "FAILED",  "time_s": round(elapsed, 1)}
            print(f"\n  [FAIL]  Step {step_num} failed  (exit code {result.returncode})")
            print("  Continuing to next step…")

    except Exception as exc:
        elapsed = time.time() - t_start
        status  = "[X]  ERROR"
        results_log[step_num] = {"status": "ERROR", "time_s": round(elapsed, 1),
                                  "error": str(exc)}
        print(f"\n  [FAIL]  Step {step_num} error: {exc}")

    print(f"\n  {status}  [{elapsed:.1f}s]")

# -------------------------------------------------------------
# FINAL SUMMARY
# -------------------------------------------------------------
header("PIPELINE SUMMARY", "=")
total_success = sum(1 for v in results_log.values() if v["status"] == "SUCCESS")
total_time    = sum(v["time_s"] for v in results_log.values())

print(f"\n  {'Step':<6} {'Name':<28} {'Status':<12} {'Time':>8}")
print(f"  {'-'*6} {'-'*28} {'-'*12} {'-'*8}")
for step_num, info in results_log.items():
    name = STEPS[step_num][0]
    status = info["status"]
    t = info["time_s"]
    icon = "[OK]" if status == "SUCCESS" else "[X]"
    print(f"  {step_num:<6} {name:<28} {icon} {status:<10} {t:>6.1f}s")

print(f"\n  Total steps run : {len(steps_ordered)}")
print(f"  Successful      : {total_success}/{len(steps_ordered)}")
print(f"  Total time      : {total_time:.1f}s  ({total_time/60:.1f} min)")

# Save run log
os.makedirs("results", exist_ok=True)
with open("results/pipeline_log.json", "w") as f:
    json.dump(results_log, f, indent=2)
print(f"\n  [OK]  Log saved -> results/pipeline_log.json")

# -------------------------------------------------------------
# LAUNCH STREAMLIT
# -------------------------------------------------------------
if args.app:
    header("LAUNCHING STREAMLIT APP", "-")
    app_path = os.path.join(ROOT, "app", "streamlit_app.py")
    print(f"  Running: streamlit run {app_path}")
    subprocess.run([PYTHON, "-m", "streamlit", "run", app_path], cwd=ROOT)
else:
    print(f"\n  [*] To launch the web app, run:")
    print(f"     python main.py --app")
    print(f"  or: streamlit run app/streamlit_app.py")

header("DONE", "=")

