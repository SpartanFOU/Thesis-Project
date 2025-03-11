from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TMP_DIR = PROJECT_ROOT / 'tmp'
DATA_DIR = PROJECT_ROOT / 'data'
LSTM_MODELS_DIR = PROJECT_ROOT / 'autoregression' /'models' / 'saved'
