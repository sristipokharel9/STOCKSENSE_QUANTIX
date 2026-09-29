"""
Phase 1 runner: cleaning + master build + Data Quality Report.
Run: python src/run_phase1.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DQ_LOG_JSON, TRAP_LOG, DQ_REPORT_MD, DQ_REPORT_CSV,
)
from data_cleaning import main as cleaning_main
from build_master  import build_master


def write_dq_report():
    with open(DQ_LOG_JSON) as f:
        log = json.load(f)

    df = pd.DataFrame(log)
    df.to_csv(DQ_REPORT_CSV, index=False)

    # detection rate vs trap log
    trap = {}
    if Path(TRAP_LOG).exists():
        with open(TRAP_LOG) as f:
            trap = json.load(f)

    md = ["# StockSense - Data Quality Report\n",
          "Every cleaning action is listed with its count, action and justification.\n",
          "## Cleaning actions\n",
          df.to_markdown(index=False),
          "\n## Injected trap log (Phase 0)\n",
          "```json",
          json.dumps(trap, indent=2),
          "```\n",
          "## Notes\n",
          "- Rejected rows are stored in `data/processed/rejected_rows.csv`.",
          "- `inventory_mismatch` is kept as an audit feature.",
          "- Sparse-history store x product combos are flagged for low-confidence predictions.",
    ]
    with open(DQ_REPORT_MD, "w") as f:
        f.write("\n".join(md))

    print(f"\nDQ report saved: {DQ_REPORT_MD}")
    print(f"DQ report CSV:   {DQ_REPORT_CSV}")


def main():
    cleaning_main()
    build_master()
    write_dq_report()
    print("\nPhase 1 COMPLETE.")


if __name__ == "__main__":
    main()