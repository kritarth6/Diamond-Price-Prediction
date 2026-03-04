import streamlit as st
import pandas as pd
import joblib
import time

# Load model
model = joblib.load("final_pipeline_model.pkl")

st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="wide"
)

# ---------------- CSS ---------------- #
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.card {
    background: rgba(255, 255, 255, 0.08);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 0px 20px rgba(0,0,0,0.3);
}
.stButton>button {
    background: linear-gradient(135deg, #00eaff, #00ff9c);
    color: black;
    font-size: 18px;
    border-radius: 12px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
st.markdown("<h1 style='text-align:center;'>💎 Diamond Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered price estimation with Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("⚙️ Enter Diamond Details")

carat = st.sidebar.number_input("Carat", min_value=0.0, step=0.01)
depth = st.sidebar.number_input("Depth", min_value=0.0, step=0.1)
table = st.sidebar.number_input("Table", min_value=0.0, step=0.1)

x = st.sidebar.number_input("X dimension", min_value=0.0, step=0.01)
y = st.sidebar.number_input("Y dimension", min_value=0.0, step=0.01)
z = st.sidebar.number_input("Z dimension", min_value=0.0, step=0.01)

cut = st.sidebar.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.sidebar.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.sidebar.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

# ---------------- MAIN ---------------- #
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📋 Your Inputs")
    st.write(f"*Carat:* {carat}")
    st.write(f"*Depth:* {depth}")
    st.write(f"*Table:* {table}")
    st.write(f"*X:* {x}")
    st.write(f"*Y:* {y}")
    st.write(f"*Z:* {z}")
    st.write(f"*Cut:* {cut}")
    st.write(f"*Color:* {color}")
    st.write(f"*Clarity:* {clarity}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎯 Prediction")
    
    if st.button("💰 Predict Price"):
        with st.spinner("Calculating price..."):
            time.sleep(1.2)

            input_data = pd.DataFrame({
                "carat": [carat],
                "depth": [depth],
                "table": [table],
                "x": [x],
                "y": [y],
                "z": [z],
                "cut": [cut],
                "color": [color],
                "clarity": [clarity]
            })

            prediction = model.predict(input_data)[0]
        
        st.success(f"💎 Estimated Price: *${prediction:,.2f}*")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.markdown("<p style='text-align:center;'>Built with ❤️ using Machine Learning | Streamlit App</p>", unsafe_allow_html=True)