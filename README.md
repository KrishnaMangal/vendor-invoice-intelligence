# Vendor Invoice Intelligence Portal 📦

An automated intelligence portal and machine learning pipeline designed to improve cost forecasting and streamline approvals by predicting freight costs and flagging abnormal vendor invoices for manual review.

## 🚀 Features

The application consists of two main Machine Learning modules:

1. **Freight Cost Prediction**
   - **Objective:** Predict the freight cost for vendor invoices based on the quantity and invoice total (Dollars) to improve cost forecasting.
   - **How it works:** Provides an estimated freight cost using a trained machine learning model, helping teams gauge standard expected shipping fees.

2. **Invoice Risk Flagging** 🚨
   - **Objective:** Classify whether a vendor invoice should be auto-approved or flagged for manual review due to abnormal patterns.
   - **How it works:** Evaluates the Purchase Order (PO) quantities and dollars against the incoming Invoice quantities, dollars, and freight cost to detect inconsistencies or high-risk claims.

## 🛠️ tech Stack

- **Python** (Core Logic & Data Processing)
- **Streamlit** (Interactive Web Dashboard UI)
- **scikit-learn** (Machine Learning models)
- **Pandas / NumPy** (Data Manipulation)
- **Matplotlib / Seaborn** (Data Visualization in Jupyter Notebooks)

## 📂 Project Structure

```
.
├── app.py                            # Main Streamlit web application
├── inference/                        # Scripts for running model predictions
│   ├── predict_freight.py            # Inference script for Freight Cost
│   └── predict_invoice_flag.py       # Inference script for Invoice Risk
├── models/                           # Serialized/saved Machine Learning models
├── data/                             # Datasets used (consider keeping local/ignored on git)
├── freight_exploration.ipynb         # EDA for Freight Prediction
├── invoice_flagging.ipynb            # EDA and Model Training for Risk Flagging
├── predicting_freight_cost.ipynb     # Model Training for Freight Prediction
├── requirements.txt                  # Python dependencies
└── .gitignore                        # Git ignore file
```

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/vendor-invoice-intelligence.git
   cd vendor-invoice-intelligence
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Running the Application

Once you have installed the required dependencies, run the Streamlit application using the following command:

```bash
streamlit run app.py
```

The application will launch in your default web browser (typically at `http://localhost:8501`).

## 🧠 Jupyter Notebooks & Model Details
Feel free to explore the Jupyter Notebooks located in the root directory. They outline the entire data science pipeline including:
- Exploratory Data Analysis (EDA)
- Feature Engineering & Data Preprocessing
- Algorithm Selection & Model Training (scikit-learn)
- Evaluation Metrics
