import streamlit as st
import pandas as pd
import pickle

# Title and Description
st.title("Car Price Prediction")
st.write("This app predicts the selling price of a car based on its features using a pre-trained model.")

# Load pre-trained model
model_file = st.file_uploader("Upload your model file (Pickle format)", type="pkl")

if model_file is not None:
    model = pickle.load(model_file)
    st.write("Model loaded successfully!")

    # Input fields for prediction
    st.write("### Predict Car Price")
    user_input = {
        'name': st.number_input("Car Brand Code", min_value=1, step=1),
        'year': st.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1),
        'km_driven': st.number_input("Kilometers Driven", min_value=0),
        'fuel': st.number_input("Fuel Type Code", min_value=1, max_value=4, step=1),
        'seller_type': st.number_input("Seller Type Code", min_value=1, max_value=3, step=1),
        'transmission': st.number_input("Transmission Code", min_value=1, max_value=2, step=1),
        'owner': st.number_input("Owner Type Code", min_value=1, max_value=5, step=1),
        'mileage': st.number_input("Mileage (kmpl)", min_value=0.0),
        'engine': st.number_input("Engine Capacity (CC)", min_value=0.0),
        'max_power': st.number_input("Max Power (bhp)", min_value=0.0),
        'seats': st.number_input("Seats", min_value=2.0, max_value=8.0, step=1.0)
    }

    # Convert user input to DataFrame
    user_data = pd.DataFrame([user_input])

    # Predict
    prediction = model.predict(user_data)
    st.write(f"### Predicted Selling Price: {prediction[0]:,.2f}")
