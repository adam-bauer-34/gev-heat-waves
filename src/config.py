from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "paths.yaml"

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

DATA_ROOT = Path(CONFIG["data_root"])
# change