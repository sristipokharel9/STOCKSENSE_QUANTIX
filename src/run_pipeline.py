"""
One command that runs the full StockSense pipeline end to end.

Usage:
    python src/run_pipeline.py

Order:
    Phase 0  simulate_hackathon_tables
    Phase 1  cleaning + master + DQ report
    Phase 2  KPIs + stats tests + EDA
    Phase 3  features + leakage test
    Phase 4  demand model
    Phase 5  stockout model
    Phase 6  explainability
    Phase 7  recommendations
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run(cmd, title):
    print("\n" + "=" * 72)
    print(f">>> {title}")
    print("=" * 72)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / cmd)],
                       cwd=str(ROOT))
    dt = time.time() - t0
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {title} (exit {r.returncode})")
    print(f"<<< {title} done in {dt:.1f}s")


def main():
    started = time.time()
    run("simulate_hackathon_tables.py", "Phase 0 - simulate 5 raw tables")
    run("run_phase1.py",               "Phase 1 - clean + master + DQ")
    run("run_phase2.py",               "Phase 2 - KPIs + stats + EDA")
    run("run_phase3.py",               "Phase 3 - features + leakage test")
    run("run_phase4.py",               "Phase 4 - demand model")
    run("run_phase5.py",               "Phase 5 - stockout model")
    run("run_phase6.py",               "Phase 6 - explainability")
    run("run_phase7.py",               "Phase 7 - recommendations")
    print("\n" + "=" * 72)
    print(f"PIPELINE COMPLETE in {time.time() - started:.1f}s")
    print("Next: streamlit run dashboard/app.py")
    print("=" * 72)


if __name__ == "__main__":
    main()