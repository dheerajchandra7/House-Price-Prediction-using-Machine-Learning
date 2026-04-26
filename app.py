import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="House Price Predictor", layout="centered")

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("model.pkl", "rb"))

# ------------------ INR FORMAT FUNCTION ------------------
def format_inr(num):
    num = int(num)
    s = str(num)
    if len(s) <= 3:
        return s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        rest = ",".join([rest[max(i-2,0):i] for i in range(len(rest), 0, -2)][::-1])
        return rest + "," + last3

def format_lakh_crore(num):
    if num >= 10000000:
        return f"{num/10000000:.2f} Cr"
    elif num >= 100000:
        return f"{num/100000:.2f} Lakh"
    else:
        return str(num)

# ------------------ TITLE ------------------
st.title("🏠 House Price Prediction App")
st.markdown("Enter details below to estimate house price")

# ------------------ INPUT SECTION ------------------
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=1000)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)

with col2:
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    age = st.number_input("Age of House", min_value=0, max_value=50, value=5)

location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])

# ------------------ ENCODING ------------------
loc_suburban = 1 if location == "Suburban" else 0
loc_urban = 1 if location == "Urban" else 0

# ------------------ PREDICTION ------------------
if st.button("🔍 Predict Price"):

    input_data = np.array([[area, bedrooms, bathrooms, age, loc_suburban, loc_urban]])
    prediction = model.predict(input_data)[0]

    # formatted outputs
    inr_price = format_inr(prediction)
    short_price = format_lakh_crore(prediction)

    st.success(f"💰 Estimated Price: ₹ {inr_price} ({short_price})")

    # ------------------ FEATURE IMPORTANCE ------------------
    st.subheader("📊 Feature Impact")

    try:
        features = ["Area", "Bedrooms", "Bathrooms", "Age", "Suburban", "Urban"]
        importance = model.coef_

        df_imp = pd.DataFrame({
            "Feature": features,
            "Impact": importance
        })

        st.dataframe(df_imp)

        # bar chart
        plt.figure()
        plt.bar(features, importance)
        plt.xticks(rotation=30)
        plt.title("Feature Importance")
        st.pyplot(plt)

    except:
        st.info("Feature importance not available for this model")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")