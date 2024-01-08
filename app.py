import streamlit as st
import pandas as pd
import requests

# Function to predict churn using FastAPI
def predict_churn_interface(TENURE, MONTANT, FREQUENCE_RECH, REVENUE, ARPU_SEGMENT, FREQUENCE,
                             DATA_VOLUME, ON_NET, ORANGE, TIGO, REGULARITY, FREQ_TOP_PACK):

    
     # Send data to FastAPI for prediction
    response = requests.post("http://127.0.0.1:8000/predict_churn", json=input_data)
    prediction_data = response.json()
    
    
# Extract probability score and churn status from the API response
    probability_score = prediction_data.get("probability_score", "N/A")
    churn_status = prediction_data.get("churn_status", "N/A")


    # Display prediction result in Streamlit
    st.write(f"Prediction: {churn_status}\nProbability Score: {probability_score}")

# Set up interface
# Inputs
input_data = {
    "TENURE": st.selectbox("What is the duration of your network?", ['I 18-21 month', 'K > 24 months', 'G 12-15 months',
                                                                     'J 21-24 months', 'H 15-18 months', 'F 9-12 months',
                                                                     'E 6-9 months', 'D 3-6 months']),
    "MONTANT": st.slider("What is your top-amount?", 15, 800, 10),
    "FREQUENCE_RECH": st.slider("What is the number of times you refilled your bundle?", 5, 200, 10),
    "REVENUE": st.slider("What is your monthly income", 100, 10000, 200),
    "ARPU_SEGMENT": st.slider("What is your income over 90 days / 3", 1000, 500000, 1000),
    "FREQUENCE": st.slider("How often do you use the service", 10, 200, 5),
    "DATA_VOLUME": st.slider("How many times do you have connections", 20, 1000, 2),
    "ON_NET": st.slider("How many times do you do inter expresso calls", 5, 1000, 3),
    "ORANGE": st.slider("How many times do you use orange to make calls (tigo)", 5, 100, 2),
    "TIGO": st.slider("How many times do you use tigo networks", 6, 100, 5),
    "REGULARITY": st.slider("How many times are you active for 90 days", 5, 100, 2),
    "FREQ_TOP_PACK": st.slider("How many times have you been activated to the top pack packages", 10, 1000, 5),
}

# Call prediction function when a button is clicked
if st.button("Predict"):
    predict_churn_interface(**input_data)
