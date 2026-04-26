# Vendor Invoice Intelligence Portal

An end-to-end ML project for vendor invoice intelligence with two modules:

1. Freight cost prediction
2. Invoice risk flagging for manual review

The project includes training pipelines, inference modules, Jupyter notebooks, and a Streamlit app.

## Features

1. Freight Cost Prediction
1. Uses Quantity and Dollars as input
1. Predicts Freight cost

1. Invoice Risk Flagging
1. Predicts if an invoice should be flagged for manual approval
1. Uses PO and invoice behavior features

## Tech Stack

1. Python
1. Streamlit
1. scikit-learn
1. Pandas
1. SQLAlchemy
1. PostgreSQL

## Project Structure

.
|- app.py
|- Invoice_Flagging.ipynb
|- Predicting_Freight_Cost.ipynb
|- requirements.txt
|- models/
|- inference/
| |- predict_freight.py
| |- predict_invoice_flag.py
|- freight_cost_prediction/
| |- data_preprocessing.py
| |- model_evaluation.py
| |- train.py
|- invoice_flagging/
| |- data_preprocessing.py
| |- model_evaluation.py
| |- train.py

## Setup

1. Clone repository
   git clone <your-repo-url>
   cd <your-project-folder>

2. Create and activate virtual environment
   python -m venv .venv

   Windows:
   .venv\Scripts\activate

   macOS/Linux:
   source .venv/bin/activate

3. Install dependencies
   pip install -r requirements.txt

## Train Models

1. Freight model
   python freight_cost_prediction/train.py

2. Invoice flag model
   python invoice_flagging/train.py

## Run App

1. Start Streamlit
   streamlit run app.py

2. Open in browser
   http://localhost:8501

If port 8501 is busy, run:
streamlit run app.py --server.port 8502

## Push to GitHub

1. Check status
   git status

2. Add files
   git add .

3. Commit
   git commit -m "Update README and project pipeline"

4. Push
   git push origin main
