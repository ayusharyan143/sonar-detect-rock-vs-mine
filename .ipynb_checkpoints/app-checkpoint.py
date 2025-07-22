import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load('logistic_model.pkl')

st.title("Sonar Object Classifier")
st.write("Enter 60 sonar features to predict Rock or Mine")

# Input fields for 60 features
features = [st.number_input(f"Feature {i+1}") for i in range(60)]

if st.button("Predict"):
    input_data = np.asarray(features).reshape(1, -1)
    prediction = model.predict(input_data)
    result = "The object is a Rock" if prediction[0] == 'R' else "The object is a Mine"
    st.success(result)
