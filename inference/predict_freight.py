import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'predict_freight_model.pkl'

def load_model(model_path):
    return joblib.load(model_path)

def predict_freight_cost(input_data):
    model = load_model(MODEL_PATH)
    input_df = pd.DataFrame([
        {
            "Dollars": input_data["Dollars"],
            "Quantity": input_data["Quantity"],
        }
    ])
    prediction = model.predict(input_df)
    return round(prediction[0], 2)