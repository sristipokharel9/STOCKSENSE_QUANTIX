"""
Phase 3 runner: features + leakage test.
Run: python src/run_phase3.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from _loader import load_sibling


def main():
    load_sibling("features").main()
    print("\nRunning leakage test...")
    subprocess.run([sys.executable, str(ROOT / "tests" / "test_no_leakage.py")], check=True)
    print("\nPhase 3 COMPLETE.")


if __name__ == "__main__":
    main()