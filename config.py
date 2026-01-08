"""Config file that loads paths from YAML.

Adam Bauer
UChicago
Jan 2026
"""

from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent / "config" / "paths.yaml"

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

DATA_ROOT = Path(CONFIG["data_root"])