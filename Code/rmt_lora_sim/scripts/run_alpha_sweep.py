from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rmt_lora.cli import run_alpha_cli

if __name__ == "__main__":
    run_alpha_cli()
