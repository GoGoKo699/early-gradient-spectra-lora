from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rmt_lora.cli import plot_cli

if __name__ == "__main__":
    plot_cli()
