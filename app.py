import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="wide"
)

# Load model
model = joblib.load("final_pipeline_model.pkl")

# Title
st.title("💎 Diamond Price Prediction App")
st.markdown("Predict diamond price using Machine Learning")

st.write("---")

# Sidebar
st.sidebar.header("Enter Diamond Features")

carat = st.sidebar.slider("Carat", 0.2, 5.0, 1.0)
depth = st.sidebar.slider("Depth", 40.0, 80.0, 60.0)
table = st.sidebar.slider("Table", 40.0, 100.0, 60.0)

x = st.sidebar.slider("Length (x)", 0.0, 10.0, 5.0)
y = st.sidebar.slider("Width (y)", 0.0, 10.0, 5.0)
z = st.sidebar.slider("Height (z)", 0.0, 10.0, 3.0)

cut = st.sidebar.selectbox(
    "Cut",
    ["Fair", "Good", "Very Good", "Premium", "Ideal"]
)

color = st.sidebar.selectbox(
    "Color",
    ["D","E","F","G","H","I","J"]
)

clarity = st.sidebar.selectbox(
    "Clarity",
    ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]
)

st.write("### Selected Features")

input_data = pd.DataFrame({
    "carat":[carat],
    "depth":[depth],
    "table":[table],
    "x":[x],
    "y":[y],
    "z":[z],
    "cut":[cut],
    "color":[color],
    "clarity":[clarity]
})

st.dataframe(input_data)

st.write("---")

if st.button("Predict Price 💎"):

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Diamond Price: **${prediction:,.2f}**")

    st.balloons()

st.write("---")

st.markdown(
"""
### About
This app predicts diamond prices using a **Machine Learning model built with Scikit-Learn Random Forest**.

Features used:
- Carat
- Depth
- Table
- Dimensions
- Cut
- Color
- Clarity
"""
)
