import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'predict_flag_invoice.pkl'
SCALER_PATH = PROJECT_ROOT / 'models' / 'scaler.pkl'

def load_model(model_path):
    return joblib.load(model_path)

def predict_invoice_flag(input_data):
    model = load_model(MODEL_PATH)
    scaler = load_model(SCALER_PATH)
    
    input_df = pd.DataFrame([input_data])
    
    # Scale exactly as during training
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    
    return bool(prediction[0])