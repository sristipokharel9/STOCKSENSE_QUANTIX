"""
Phase 2 runner: KPIs + stats tests + EDA figures.
Run: python src/run_phase2.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _loader import load_sibling


def main():
    load_sibling("kpis").main()
    load_sibling("stats_tests").main()
    load_sibling("eda").main()
    print("\nPhase 2 COMPLETE.")


if __name__ == "__main__":
    main()