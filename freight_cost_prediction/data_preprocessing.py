import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

def load_vendor_invoice_data():
    db_url = os.getenv("DATABASE_URL")
    if db_url and db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    engine = create_engine(db_url)
    # Extracting the required columns for regression
    query = 'SELECT "Quantity", "Dollars", "Freight" FROM "vendor_invoice"'
    df = pd.read_sql_query(query, engine)
    return df

def prepare_features(df):
    # Using Dollars and Quantity to predict Freight
    X = df[['Dollars', 'Quantity']]
    y = df['Freight']
    return X, y

def split_data(X, y):
    # Splitting with 20% test size and random state 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
