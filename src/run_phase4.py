import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _loader import load_sibling


def main():
    load_sibling("train_forecast").main()
    print("\nPhase 4 COMPLETE.")


if __name__ == "__main__":
    main()