import streamlit as st
import pandas as pd
import sys
import os

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag

st.set_page_config(page_title="Vendor Invoice Intelligence Portal", layout="wide")

st.header("Vendor Invoice Intelligence Portal 📦")
st.markdown("Automated intelligence for freight cost estimation and risk flagging.")
st.divider()

# Sidebar Navigation
st.sidebar.title("Model Selection")
selected_model = st.sidebar.radio("Choose Prediction Module", ["Freight Cost Prediction", "Invoice Risk Flagging"])

if selected_model == "Freight Cost Prediction":
    st.subheader("Predicting Freight Cost")
    st.markdown("Objective: Predict freight cost for vendor invoice using quantity and invoice dollars to improve cost forecasting.")

    col1, col2 = st.columns(2)
    with col1:
        quantity = st.number_input("Quantity", min_value=1, value=100)
    with col2:
        dollars = st.number_input("Invoice Dollars ($)", min_value=1.0, value=500.0)
        
    if st.button("Submit Freight"):
        input_data = {"Quantity": quantity, "Dollars": dollars}
        prediction = predict_freight_cost(input_data)
        st.success(f"Predicted Freight Cost: ${prediction}")

elif selected_model == "Invoice Risk Flagging":
    st.subheader("Flagging Vendor Invoices for Manual Review")
    st.markdown("Objective: Predict whether a vendor invoice should be flagged for manual approval based on abnormal patterns.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_item_quantity = st.number_input("Total Item Quantity (PO)", value=100)
        invoice_quantity = st.number_input("Invoice Quantity", value=100)
    with col2:
        total_item_dollars = st.number_input("Total Item Dollars (PO)", value=500.0)
        invoice_dollars = st.number_input("Invoice Dollars", value=500.0)
    with col3:
        freight = st.number_input("Freight Cost", value=20.0)
        
    if st.button("Submit For Risk Flagging"):
        input_data = {
            "Total_Item_Quantity": total_item_quantity,
            "Total_Item_Dollars": total_item_dollars,
            "Invoice_Quantity": invoice_quantity,
            "Invoice_Dollars": invoice_dollars,
            "Freight": freight
        }
        is_flagged = predict_invoice_flag(input_data)
        
        if is_flagged:
            st.error("🚨 Invoice Requires Manual Approval (Abnormal Pattern Detected)")
        else:
            st.success("✅ Invoice is Safe for Auto Approval")