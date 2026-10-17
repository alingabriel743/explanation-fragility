"""Loading of the three benchmark datasets."""
import os
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get("FRAGILITY_DATA", os.path.join(ROOT, "data"))

SOURCES = {
    "telco": ("Telco-Customer-Churn.csv",
              "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"),
    "german_credit": ("german_credit.csv",
                      "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"),
    "hr_attrition": ("HR-Employee-Attrition.csv",
                     "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"),
}
DISPLAY = {"telco": "Telco Customer Churn", "german_credit": "German Credit Risk", "hr_attrition": "IBM HR Attrition"}

GERMAN_COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount", "savings_status",
    "employment", "installment_rate", "personal_status", "other_parties", "residence_since",
    "property_magnitude", "age", "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker", "class",
]


def download(name):
    fname, url = SOURCES[name]
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def load(name):
    """Return features X, binary target y and the display name of a dataset."""
    path = download(name)
    if name == "telco":
        df = pd.read_csv(path)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.dropna(subset=["Churn"])
        X = df.drop(columns=["customerID", "Churn"])
        y = df["Churn"].map({"Yes": 1, "No": 0})
    elif name == "german_credit":
        df = pd.read_csv(path, sep=" ", header=None, names=GERMAN_COLUMNS)
        X = df.drop(columns=["class"])
        y = (df["class"] == 2).astype(int)          # 2 = bad credit risk
    elif name == "hr_attrition":
        df = pd.read_csv(path)
        constant = [c for c in ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"] if c in df.columns]
        X = df.drop(columns=["Attrition"] + constant)
        y = df["Attrition"].map({"Yes": 1, "No": 0})
    else:
        raise ValueError(name)
    return X.reset_index(drop=True), y.reset_index(drop=True).astype(int), DISPLAY[name]
