import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_model():
    # Load the model, scaler, and label encoder from the pickle file
    with open("student_linearREG_final_model.pkl", 'rb') as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le


def preprocess_input(data, scaler, le):
    # Create DataFrame with correct feature order
    features = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities','Sleep Hours', 'Sample Question Papers Practiced']
    df = pd.DataFrame([data], columns=features)
    df['Extracurricular Activities'] = le.transform(df['Extracurricular Activities'])      # Encode categorical feature
    
    # Scale features
    return scaler.transform(df)


def predict_data(data):  # Accept input data as a parameter
    # Load model resources
    model, scaler, le = load_model()
    processed_data = preprocess_input(data, scaler, le)      # Preprocess the input data
    prediction = model.predict(processed_data)   # Make predictions using the loaded model
    return prediction


def main():
    st.title("Student Performance Prediction")
    st.write("Enter student details to predict their performance:")

    # Input fields for user data
    hours_studied = st.number_input("Hours studied", min_value=1, max_value=10, value=5)
    previous_score = st.number_input("Previous score", min_value=40, max_value=100, value=50)
    extra_activity = st.selectbox("Extra curricular Activity", ['Yes', 'No'])
    sleeping_hours = st.number_input("Sleeping Hours", min_value=3, max_value=10, value=8)
    questions_solved = st.number_input("Number of questions solved", min_value=0, max_value=10, value=5)

    if st.button("Predict your score"):
        # Map user inputs to the expected feature names
        user_data = {
            "Hours Studied": hours_studied,
            "Previous Scores": previous_score,
            "Extracurricular Activities": extra_activity,
            "Sleep Hours": sleeping_hours,
            "Sample Question Papers Practiced": questions_solved
        }
        
        # Call predict_data with user data
        prediction = predict_data(user_data)
        
        # Display the prediction result
        st.success(f"Your predicted performance score is: {prediction[0]:.2f}")


if __name__ == "__main__":
    main()
