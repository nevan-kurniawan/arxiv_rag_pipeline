from pathlib import Path

ROOT = Path(__file__).parent.parent

DATA_DIR = ROOT / "data"
RAW_DATA = DATA_DIR / "raw"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
EVAL_DIR = DATA_DIR / "evaluation"
MODEL_CONFIG_DIR = ROOT / "config" / "model_configs.yaml"
RESULTS_DIR = ROOT / "results"