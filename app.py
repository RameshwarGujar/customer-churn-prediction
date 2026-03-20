import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle

# Load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Page setup
st.set_page_config(page_title="Customer Churn App", layout="wide")
load_css("styles.css")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_excel("Telecom_Customer_Churn_Analysis.xlsx", engine='openpyxl')
    return df

df = load_data()
df['Churn_encoded'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Sidebar Filters
st.markdown("<div class='big-title'>Telecom Customer Churn Analysis</div>", unsafe_allow_html=True)
st.sidebar.header("🔎 Filter Options")
gender_filter = st.sidebar.selectbox("Select Gender", options=["All"] + sorted(df['gender'].unique()))
contract_filter = st.sidebar.selectbox("Select Contract Type", options=["All"] + sorted(df['Contract'].unique()))

filtered_df = df.copy()
if gender_filter != "All":
    filtered_df = filtered_df[filtered_df['gender'] == gender_filter]
if contract_filter != "All":
    filtered_df = filtered_df[filtered_df['Contract'] == contract_filter]

# Train Model
@st.cache_resource
def train_model(data):
    data = data.copy()
    le_gender = LabelEncoder()
    le_contract = LabelEncoder()
    data['gender_encoded'] = le_gender.fit_transform(data['gender'])
    data['contract_encoded'] = le_contract.fit_transform(data['Contract'])
    X = data[['gender_encoded', 'tenure', 'MonthlyCharges', 'contract_encoded']]
    y = data['Churn_encoded']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, le_gender, le_contract

model, le_gender, le_contract = train_model(df)

# Prediction Logic
def make_prediction(input_data, model):
    input_df = pd.DataFrame([input_data])
    input_df['gender_encoded'] = le_gender.transform(input_df['gender'])
    input_df['contract_encoded'] = le_contract.transform(input_df['Contract'])
    X = input_df[['gender_encoded', 'tenure', 'MonthlyCharges', 'contract_encoded']]
    prediction = model.predict(X)[0]
    return "Churn" if prediction == 1 else "Not Churn"

# Tabs


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Overview", "🔍 Data Preview", "📈 EDA + ML", "📊 Power BI Dashboard", "🤖 Churn Prediction"
])

# Overview Tab
# Overview
with tab1:
    st.markdown("<h1 class='white-header'>Customer Churn Analysis</h1>", unsafe_allow_html=True)

    st.markdown("🔍 Problem Statement")
    st.write("""
    Telecom companies often struggle with customer retention. Churned customers result in lost revenue and higher acquisition costs. 
    This project focuses on identifying patterns and building a predictive model to flag customers who are likely to churn.
    """)

    st.markdown(" 🎯 Objectives")
    st.markdown("""
    - Identify key drivers of customer churn.
    - Visualize trends and patterns in the data.
    - Build a machine learning model to predict churn.
    - Integrate interactive dashboards and prediction tools for business use.
    """)

    st.markdown(" 📊 Dataset Information")
    st.markdown("""
    - Rows: ~7000+ customer records
    - Columns: 21 features including demographic data, services, account info, etc.
    - Target Variable: `Churn` (Yes/No)
    """)

    st.markdown(" 🛠️ Technologies Used")
    st.markdown("""
    - Python
    - Streamlit (for UI)
    - Pandas, Seaborn, Matplotlib** (for analysis & plots)
    - Scikit-learn** (for modeling)
    - Power BI** (for professional dashboards)
    """)

    st.markdown("🧠 Project Workflow")
    st.image("C:/Users/rames/OneDrive/Desktop/Churn_ui_project/Workflow_Diagram.png", caption="End-to-End Flow", width=500)

    st.markdown("📌 Business Impact")
    st.write("""
    - Helps the business proactively retain high-risk customers.
    - Provides customer insight and aids strategic planning.
    - Saves on acquisition costs by improving retention.
    """)

# Data Preview Tab
with tab2:
    st.header("Dataset Preview")
    st.dataframe(filtered_df)  # Display all rows


# EDA + ML Tab
with tab3:
    st.header("Exploratory Data Analysis & Model Evaluation")

    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x='Churn', ax=ax)
    st.pyplot(fig)

    st.subheader("Contract vs Churn")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x='Contract', hue='Churn', ax=ax)
    plt.xticks(rotation=20)
    st.pyplot(fig)

    st.subheader("Gender vs Churn")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x='gender', hue='Churn', ax=ax)
    st.pyplot(fig)

    st.subheader("Monthly Charges Distribution by Churn")
    fig, ax = plt.subplots()
    sns.histplot(data=filtered_df, x='MonthlyCharges', hue='Churn', kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Tenure Distribution by Churn")
    fig, ax = plt.subplots()
    sns.histplot(data=filtered_df, x='tenure', hue='Churn', kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Boxplot - Monthly Charges vs Churn")
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_df, x='Churn', y='MonthlyCharges', ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    corr = filtered_df[['tenure', 'MonthlyCharges', 'Churn_encoded']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Model Evaluation Report")
    X = filtered_df[['gender', 'tenure', 'MonthlyCharges', 'Contract']]
    X.loc[:, 'gender_encoded'] = le_gender.transform(X['gender'])
    X.loc[:, 'contract_encoded'] = le_contract.transform(X['Contract'])
    X = X[['gender_encoded', 'tenure', 'MonthlyCharges', 'contract_encoded']]
    y = filtered_df['Churn_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))


# Power BI Dashboard
with tab4:
    st.header("Power BI Dashboard")
    st.components.v1.iframe(
        src="https://app.powerbi.com/view?r=eyJrIjoiXXXXXXXXXXXX",  
        width=1200,
        height=800,
        scrolling=True
    )

# Churn Prediction
with tab5:
    st.header("Churn Prediction")
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.slider("Tenure", 0, 72)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    if st.button("Predict"):
        input_data = {
            'gender': gender,
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'Contract': contract
        }
        prediction = make_prediction(input_data, model)
        st.success(f"Prediction: {prediction}")
