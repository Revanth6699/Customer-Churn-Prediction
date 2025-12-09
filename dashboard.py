import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

st.title('Customer Churn Prediction Dashboard')
model = pickle.load(open(r"C:\Users\Windows\Desktop\customer\xgb_churn_model.pkl", 'rb'))  # Update path as needed!

uploaded_file = st.file_uploader('Upload customer CSV', type='csv')
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Drop 'customerID' if present
    if 'customerID' in df.columns:
        df = df.drop(['customerID'], axis=1)

    # Convert 'TotalCharges' to numeric and fill missing
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # Encode categorical columns (except Churn, if present)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'Churn':
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # If there is a 'Churn' column in uploaded data, drop it (since this is for prediction)
    if 'Churn' in df.columns:
        df = df.drop(['Churn'], axis=1)

    # ==== FEATURE ENGINEERING AS IN TRAINING ====
    # Tenure bucket
    if 'tenure' in df.columns:
        df['tenure_bucket'] = pd.cut(df['tenure'], bins=[-np.inf, 12, 24, np.inf], labels=['Short', 'Medium', 'Long'])
        df['tenure_bucket'] = LabelEncoder().fit_transform(df['tenure_bucket'].astype(str))
    else:
        df['tenure_bucket'] = 0  # fallback if missing

    # Engagement ratio
    if all(x in df.columns for x in ['MonthlyCharges', 'tenure']):
        df['engagement_ratio'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    else:
        df['engagement_ratio'] = 0  # fallback if missing

    # Complaint rate (dummy zero, since not present in input CSV)
    df['complaint_rate'] = 0

    # ==== END FEATURE ENGINEERING ====

    # Predict churn probabilities
    pred_prob = model.predict_proba(df)[:, 1]
    df['Churn Probability'] = pred_prob

    st.write(df[['Churn Probability']])

    # Matplotlib bar chart (fixes Altair/Streamlit chart issue)
    fig, ax = plt.subplots()
    ax.bar(df.index, df['Churn Probability'])
    ax.set_xlabel("Customer Index")
    ax.set_ylabel("Churn Probability")
    ax.set_title("Churn Probability per Customer")
    st.pyplot(fig)

    st.success("Prediction complete! Use the 'Churn Probability' column to identify high-risk customers.")

else:
    st.info("Please upload a customer CSV file. Columns must match your model features.")
