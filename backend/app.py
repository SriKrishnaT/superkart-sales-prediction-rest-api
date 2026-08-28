# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
super_kart_sales_predictor_api = Flask("Super Kart Predictor")

# Load the trained machine learning model
model = joblib.load("super_kart_sales_prediction_model_v1_0.joblib")

# Define a route for the home page (GET request)
@super_kart_sales_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the Super Kart Sales Prediction API!"

# Define an endpoint for single property prediction (POST request)
@super_kart_sales_predictor_api.post('/v1/predict')
def predict_sales():
    """
    This function handles POST requests to the '/v1/predict' endpoint.
    It expects a JSON payload containing product and store details and returns
    the predicted sales as a JSON response.
    """
    # Get the JSON data from the request body
    property_data = request.get_json()

    # Calculate Store_Age from Store_Establishment_Year
    store_age = 2026 - property_data['Store_Establishment_Year']

    # Handle 'reg' in Product_Sugar_Content if present (frontend should ideally send clean data)
    product_sugar_content = property_data['Product_Sugar_Content']
    if product_sugar_content == 'reg':
        product_sugar_content = 'Regular'

    # Construct a DataFrame matching the model's expected input features
    sample = {
        'Product_Weight': property_data['Product_Weight'],
        'Product_Sugar_Content': product_sugar_content,
        'Product_Allocated_Area': property_data['Product_Allocated_Area'],
        'Product_Type': property_data['Product_Type'],
        'Product_MRP': property_data['Product_MRP'],
        'Store_Size': property_data['Store_Size'],
        'Store_Location_City_Type': property_data['Store_Location_City_Type'],
        'Store_Type': property_data['Store_Type'],
        'Store_Age': store_age
    }

    input_data = pd.DataFrame([sample])

    # Make prediction
    predicted_sales = model.predict(input_data)[0]
    # Ensure predictions are non-negative
    predicted_sales = np.maximum(0, predicted_sales)

    # Return the predicted sales, rounded for better display
    return jsonify({'Predicted Sales (in dollars)': round(float(predicted_sales), 2)})


# Define an endpoint for batch prediction (POST request)
@super_kart_sales_predictor_api.post('/v1/predictbatch')
def predict_sales_batch():
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint.
    It expects a CSV file containing details for multiple products/stores
    and returns the predicted sales as a JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Apply feature engineering
    input_data['Store_Age'] = 2026 - input_data['Store_Establishment_Year']
    # Apply data cleaning
    input_data['Product_Sugar_Content'] = input_data['Product_Sugar_Content'].replace('reg','Regular')

    # Select and order columns to match the model's expected input features
    features_for_prediction = input_data[[
        'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area',
        'Product_Type', 'Product_MRP', 'Store_Size',
        'Store_Location_City_Type', 'Store_Type', 'Store_Age'
    ]]

    # Make predictions
    predicted_sales = model.predict(features_for_prediction)
    # Ensure predictions are non-negative
    predicted_sales = np.maximum(0, predicted_sales).tolist()

    # Return the predictions list, rounded for better display
    return jsonify({'Predicted Sales (in dollars)': [round(float(s), 2) for s in predicted_sales]})

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    super_kart_sales_predictor_api.run(debug=True)
