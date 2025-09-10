import streamlit as st
import pandas as pd
import pickle
import os
import gdown

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Fund Allocation Predictor",
    page_icon="💰",
    layout="wide"
)

# -------------------------------
# Model download settings
# -------------------------------

MODEL_PATH = "Fund_Allocation.pk1"
MODEL_URL = "https://drive.google.com/uc?id=14BqODxR2fajbKlNVpoihZqYUBQb1Bzt7&export=download"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model... This may take several minutes for large files (~687MB).")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)


# -------------------------------
# Load the model
# -------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()
st.success("Model loaded successfully!")

# -------------------------------
# App Title and Description
# -------------------------------
st.title("Fund Allocation Predictor")
st.markdown("""
This application predicts the optimal allocation of funds across different investment categories 
based on your financial profile and investment preferences.
""")

# -------------------------------
# Input Columns
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial Information")
    income = st.number_input("Annual Income ($)", min_value=10000, max_value=10000000, value=100000, step=5000)
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    net_worth = st.number_input("Net Worth ($)", min_value=0, max_value=50000000, value=500000, step=10000)

with col2:
    st.subheader("Investment Preferences")
    risk_appetite = st.selectbox("Risk Appetite", ["Low", "Medium", "High"], index=1)
    investment_horizon = st.slider("Investment Horizon (years)", min_value=1, max_value=40, value=10, step=1)
    expected_return = st.slider("Expected Return (%)", min_value=1.0, max_value=20.0, value=7.0, step=0.1)
    portfolio_volatility = st.slider("Acceptable Portfolio Volatility (%)", min_value=1.0, max_value=25.0, value=10.0, step=0.1)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Fund Allocation"):
    input_data = pd.DataFrame([[
        income, age, risk_appetite, net_worth, investment_horizon, expected_return, portfolio_volatility
    ]], columns=[
        'Income',
        'Age',
        'Risk_Appetite',
        'Net_Worth',
        'Investment_Horizon',
        'Expected_Return',
        'Portfolio_Volatility'
    ])
    
    prediction = model.predict(input_data)
    allocation_columns = [
        'Stock_Allocation',
        'Bond_Allocation',
        'FD_Allocation',
        'ETF_Allocation',
        'Cash_Allocation',
        'Other_Allocation'
    ]
    prediction_df = pd.DataFrame(prediction, columns=allocation_columns)
    
    st.subheader("Recommended Fund Allocation")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        chart_data = pd.DataFrame({
            'Category': allocation_columns,
            'Percentage': prediction_df.iloc[0].values.tolist()
        })
        st.bar_chart(chart_data.set_index('Category'))
    
    with col2:
        st.dataframe(
            pd.DataFrame({
                'Category': allocation_columns,
                'Allocation (%)': [f"{value:.2f}%" for value in prediction_df.iloc[0].values]
            }).set_index('Category'),
            hide_index=False
        )
    
    st.subheader("Allocation Summary")
    total_allocation = sum(prediction_df.iloc[0].values)
    st.write(f"""
    Based on your financial profile and investment preferences, we recommend the following allocation:
    
    - **Stocks**: {prediction_df.iloc[0]['Stock_Allocation']:.2f}% - Higher risk, higher potential returns
    - **Bonds**: {prediction_df.iloc[0]['Bond_Allocation']:.2f}% - Medium risk, stable returns
    - **Fixed Deposits**: {prediction_df.iloc[0]['FD_Allocation']:.2f}% - Low risk, guaranteed returns
    - **ETFs**: {prediction_df.iloc[0]['ETF_Allocation']:.2f}% - Diversified investment with moderate risk
    - **Cash**: {prediction_df.iloc[0]['Cash_Allocation']:.2f}% - Liquid assets for emergencies
    - **Other Investments**: {prediction_df.iloc[0]['Other_Allocation']:.2f}% - Alternative investments
    
    Total allocation: {total_allocation:.2f}%
    """)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("Fund Allocation Predictor - Powered by Machine Learning")
