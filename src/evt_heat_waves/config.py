"""Config file that loads YAMLs and stores useful dictionaries / constants.

Adam Bauer
UChicago
Jan 2026

Last edited: 4/30/2026, 5:03 PM CST
"""

import yaml

from pathlib import Path

# unpack config file
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "paths.yaml"
EXT_DATA_PATH = Path(__file__).parent.parent.parent / "config" / "events_feat.yaml"

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

# directories
DATA_ROOT = Path(CONFIG["DATA_ROOT"])

FIGS_PATH = Path(CONFIG["FIGS_PATH"])
CHECKS_PATH = Path(CONFIG["CHECKS_PATH"])
ERA5_PATH = DATA_ROOT / CONFIG["ERA5_DIR"]
CMIP_PATH = DATA_ROOT / CONFIG["CMIP_DIR"]
AMIP_PATH = DATA_ROOT / CONFIG["AMIP_DIR"]
STATS_PATH = DATA_ROOT / CONFIG["STATS_DIR"]

# mapping for args.data -> data path
MIP_FIT_PATH_DICT = {
    'cmip': {
        'data': CMIP_PATH,
        'config': {
            'meta': CONFIG_PATH.parent / "meta.yaml",
            'qc': CONFIG_PATH.parent / "qc.yaml"}
    },
    'amip': {
        'data': AMIP_PATH,
        'config': {
            'meta': CONFIG_PATH.parent / "meta_amip.yaml",
            'qc': CONFIG_PATH.parent / "qc_amip.yaml"}
    },
}

# mapping for data -> variable name in dataset
ANOM_TYPE_TO_VAR = {
    'raw': 'tas',
    'annmean': 't2m_anom_annmean',
    'trend': 't2m_anom_trend'
}

# MLE characteristics
MLE_CONFIG_PATH = Path(__file__).parent.parent.parent / 'config' / 'mle_attrs.yaml'

with open(MLE_CONFIG_PATH, 'r') as f:
    MLE_ATTRS = yaml.safe_load(f)

MLE_FIT_ATTRS = MLE_ATTRS['fit']

# canonical full parameter order the MLE always solves over
MLE_FULL_PARAM_NAMES = ['loc', 'loc_t', 'scale', 'scale_t', 'shape', 'shape_t']