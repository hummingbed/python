import streamlit as st
import pandas as pd
import pickle

# Title and Description
st.title("Car Price Prediction")
st.write("This app predicts the selling price of a car based on its features using a pre-trained model.")

# Load pre-trained model
model_path = 'model.pkl'
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    st.write("Model loaded successfully!")

    # Define categorical options based on initial dataset
    name_options = ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel']
    fuel_options = ['Diesel', 'Petrol', 'LPG', 'CNG']
    seller_type_options = ['Individual', 'Dealer', 'Trustmark Dealer']
    transmission_options = ['Manual', 'Automatic']
    owner_options = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car']

    # Input fields for prediction
    st.write("### Predict Car Price")
    user_input = {
        'name': st.selectbox("Car Model", options=name_options),
        'year': st.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1, value=2014),
        'km_driven': st.number_input("Kilometers Driven", min_value=0, value=145500),
        'fuel': st.selectbox("Fuel Type", options=fuel_options),
        'seller_type': st.selectbox("Seller Type", options=seller_type_options),
        'transmission': st.selectbox("Transmission", options=transmission_options),
        'owner': st.selectbox("Owner Type", options=owner_options),
        'mileage': st.text_input("Mileage (e.g., 23.4 kmpl)", value="23.4 kmpl"),
        'engine': st.text_input("Engine Capacity (e.g., 1248 CC)", value="1248 CC"),
        'max_power': st.text_input("Max Power (e.g., 74 bhp)", value="74 bhp"),
        'seats': st.number_input("Seats", min_value=2.0, max_value=8.0, step=1.0, value=5.0)
    }

    # Process user input to match model requirements
    def clean_numeric(value, unit):
        return float(value.replace(unit, '').strip()) if unit in value else 0.0

    user_data = pd.DataFrame([{
        'name': name_options.index(user_input['name']) + 1,
        'year': user_input['year'],
        'km_driven': user_input['km_driven'],
        'fuel': fuel_options.index(user_input['fuel']) + 1,
        'seller_type': seller_type_options.index(user_input['seller_type']) + 1,
        'transmission': transmission_options.index(user_input['transmission']) + 1,
        'owner': owner_options.index(user_input['owner']) + 1,
        'mileage': clean_numeric(user_input['mileage'], 'kmpl'),
        'engine': clean_numeric(user_input['engine'], 'CC'),
        'max_power': clean_numeric(user_input['max_power'], 'bhp'),
        'seats': user_input['seats']
    }])

    # Predict
    prediction = model.predict(user_data)
    st.write(f"### Predicted Selling Price: {prediction[0]:,.2f}")

except FileNotFoundError:
    st.error("Model file not found. Please ensure 'model.pkl' exists in the directory.")
