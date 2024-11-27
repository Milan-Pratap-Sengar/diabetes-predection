import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

data = pd.read_csv(r"./diabetes_prediction_dataset.csv")
data["gender"] = data["gender"].map({"Male": 1, "Female": 2, "Other": 3})
data["smoking_history"] = data["smoking_history"].map({
    "never": 1, "No Info": 2, "current": 3, "former": 4, "ever": 5, "not current": 6
})

y = data['diabetes']
x = data.drop("diabetes", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

st.markdown("""
    <style>
    body {
        background-color: #f0f4f8; /* Light blue-gray background */
        font-family: 'Helvetica Neue', sans-serif; /* Change font for a modern look */
    }
    h1, h2, h3 {
        color: #333; /* Dark gray text for better contrast */
        text-align: center;
        font-weight: bold; /* Bold text for headings */
    }
    .stButton>button {
        background-color: #007bff; /* Bright blue background for buttons */
        color: white;
        border-radius: 4rem; /* More rounded corners */
        width: 100%;
        padding: 12px; /* Increased padding for better touch targets */
        font-size: 16px; /* Slightly larger font size */
        border: none; /* Remove default border */
        transition: background-color 0.3s ease; /* Smooth transition on hover */
    }
    .stButton>button:hover {
        background-color: #0056b3; /* Darker blue on hover */
    }
    .form-section {
        background-color: #ffffff; /* White background for form */
        padding: 30px; /* Increased padding for a spacious look */
        border-radius: 12px; /* More rounded corners */
        box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.2); /* Deeper shadow for elevation effect */
        margin: 20px; /* Add margin around form */
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌟 Diabetes Prediction App")
st.markdown("""
    Welcome to the **Diabetes Prediction App**.  
    Fill in your details below to predict the risk of diabetes.  
    Use the sliders and dropdowns to input your data, and click **Predict** to see the result.
""")

st.markdown("<div class='form-section'>", unsafe_allow_html=True)
with st.form("prediction_form"):
    st.markdown("<h3>Enter Your Health Details</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 0, 100, 25, help="Your current age in years.")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], help="Select your gender.")
        hypertension = st.selectbox("Hypertension", [0, 1], help="0: No, 1: Yes")
        bmi = st.slider("BMI", 10.0, 50.0, 25.0, step=0.1, help="Your Body Mass Index.")

    with col2:
        smoking_history = st.selectbox(
            "Smoking History",
            ["Never", "No Info", "Current", "Former", "Ever", "Not Current"],
            help="Your smoking habits history."
        )
        heart_disease = st.selectbox("Heart Disease", [0, 1], help="0: No, 1: Yes")
        hba1c = st.slider("HbA1c Level", 4.0, 15.0, 5.5, step=0.1, help="Your average blood glucose level over 3 months.")
        blood_glucose = st.slider("Blood Glucose Level", 50, 250, 100, help="Your current blood glucose level.")

    submit_button = st.form_submit_button(label="Predict")

st.markdown("</div>", unsafe_allow_html=True)


if submit_button:
    gender_map = {"Male": 1, "Female": 2, "Other": 3}
    smoking_map = {"Never": 1, "No Info": 2, "Current": 3, "Former": 4, "Ever": 5, "Not Current": 6}
    input_data = np.array([[age, gender_map[gender], smoking_map[smoking_history], hypertension, heart_disease, bmi, hba1c, blood_glucose]])

    prediction = rf_model.predict(input_data)
    result = "High risk of diabetes." if prediction[0] == 1 else "Low risk of diabetes."
    color = "red" if prediction[0] == 1 else "green"

    st.markdown(f"<h3 style='text-align: center; color: {color};'>{result}</h3>", unsafe_allow_html=True)


st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = x.columns

plt.figure(figsize=(8, 6))
plt.title("Feature Importance", fontsize=16)
plt.bar(range(x.shape[1]), importances[indices], align="center", color="#629584")
plt.xticks(range(x.shape[1]), features[indices], rotation=45, ha='right', fontsize=10)
plt.tight_layout()

st.pyplot(plt)