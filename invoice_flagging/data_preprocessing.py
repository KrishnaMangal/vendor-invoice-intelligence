import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

load_dotenv()

def load_invoice_data():
    db_url = os.getenv("DATABASE_URL")
    if db_url and db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    engine = create_engine(db_url)
    
    # Purchases table aggregation
    query_purchases = """
    SELECT 
        p."PONumber",
        COUNT(DISTINCT p."Brand") AS "Total_Brands",
        SUM(p."Quantity") AS "Total_Item_Quantity",
        SUM(p."Dollars") AS "Total_Item_Dollars",
        AVG(p."ReceivingDate"::date - p."PODate"::date) AS "Average_Receiving_Delay"
    FROM "purchases" p
    GROUP BY p."PONumber"
    """
    df_purchases = pd.read_sql_query(query_purchases, engine)
    
    # Vendor invoice query
    query_invoice = """
    SELECT 
        vi."PONumber",
        vi."Quantity" AS "Invoice_Quantity",
        vi."Dollars" AS "Invoice_Dollars",
        vi."Freight",
        (vi."InvoiceDate"::date - vi."PODate"::date) AS "Days_PO_to_Invoice",
        (vi."PayDate"::date - vi."InvoiceDate"::date) AS "Days_to_Pay"
    FROM "vendor_invoice" vi
    """
    df_invoice = pd.read_sql_query(query_invoice, engine)
    
    # Merge on PONumber
    df_merged = pd.merge(df_purchases, df_invoice, on='PONumber', how='inner')
    return df_merged

def create_invoice_risk_label(row):
    # Notebook rule: large invoice dollar mismatch or high receiving delay.
    if abs(row["Invoice_Dollars"] - row["Total_Item_Dollars"]) > 5:
        return 1
    if row["Average_Receiving_Delay"] > 10:
        return 1
    return 0

def apply_labels(df):
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
    return df

def split_data(df, features, target):
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def scale_features(X_train, X_test, scaler_path):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, 'models/scaler.pkl')
    return X_train_scaled, X_test_scaled
