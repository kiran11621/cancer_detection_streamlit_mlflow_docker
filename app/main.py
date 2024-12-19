import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("model.pkl")  # Load the trained model
scaler = joblib.load("scaler.pkl")  # Load the scaler

# Function to get user input for all 30 features
def user_input_features():
    feature_names = [
        'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10',
        'feature11', 'feature12', 'feature13', 'feature14', 'feature15', 'feature16', 'feature17', 'feature18', 'feature19', 'feature20',
        'feature21', 'feature22', 'feature23', 'feature24', 'feature25', 'feature26', 'feature27', 'feature28', 'feature29', 'feature30'
    ]
    
    features = []
    for feature_name in feature_names:
        feature_value = st.number_input(f'Enter {feature_name}')
        features.append(feature_value)
    
    return np.array(features).reshape(1, -1)

# Get the user input and scale it
features = user_input_features()

# Ensure features have the correct shape (1, 30)
if features.shape[1] != 30:
    st.error("Please provide all 30 features for prediction.")
else:
    # Scale the input features
    scaled_features = scaler.transform(features)  # Apply scaling to the input features

    # Make a prediction
    prediction = model.predict(scaled_features)
    st.write(f"Prediction: {'Malignant' if prediction == 1 else 'Benign'}")
