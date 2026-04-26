import joblib
from data_preprocessing import load_invoice_data, apply_labels, split_data, scale_features
from model_evaluation import train_random_forest_classifier, evaluate_classifier

FEATURES = [
    "Total_Item_Quantity",
    "Total_Item_Dollars",
    "Invoice_Quantity",
    "Invoice_Dollars",
    "Freight",
]

TARGET = "flag_invoice"


def main():
    # Load data
    df = load_invoice_data()
    df = apply_labels(df)

    # Prepare data
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, "models/scaler.pkl")

    # Train and evaluate model
    best_model = train_random_forest_classifier(X_train_scaled, y_train)
    evaluate_classifier(best_model, X_test_scaled, y_test)

    # Save best model
    joblib.dump(best_model, "models/predict_flag_invoice.pkl")

if __name__ == "__main__":
    main()
