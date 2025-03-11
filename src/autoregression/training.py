from src.utils.paths import TMP_DIR
import joblib

data = joblib.load(TMP_DIR / 'file_i_need.pkl')
