import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# # MongoDB connection using Streamlit secrets
# uri = st.secrets["mongo"]["uri"]
# client = MongoClient(uri, server_api=ServerApi('1'))
# db = client['student']
# collection = db['student_prediction']

@st.cache_resource
def load_model():
    with open("student_linearREG_final_model.pkl", 'rb') as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le

def preprocess_input(data, scaler, le):
    # Create DataFrame with correct feature order
    features = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities',
                'Sleep Hours', 'Sample Question Papers Practiced']
    df = pd.DataFrame([data], columns=features)
    
    # Encode categorical feature
    df['Extracurricular Activities'] = le.transform(df['Extracurricular Activities'])
    
    # Scale features
    return scaler.transform(df)

def main():
    # Load model resources once
    model, scaler, le = load_model()

    # Page styling
    st.markdown("""
    <h1 style='text-align: center; color: #808080;
               font-family: Arial; text-shadow: 2px 2px 4px #000000;
               border-bottom: 2px solid #3366FF; padding-bottom: 10px;'>
        Student Performance Prediction
    </h1>
    """, unsafe_allow_html=True)

    # Input widgets
    with st.form("prediction_form"):
        hours_studied = st.number_input("Hours studied", 1, 10, 5)
        previous_score = st.number_input("Previous score", 40, 100, 70)
        extracurricular = st.selectbox("Extracurricular activities", ['Yes', 'No'])
        sleep_hours = st.number_input("Sleeping hours", 4, 10, 7)
        papers_solved = st.number_input("Number of question papers solved", 0, 10, 5)
        
        if st.form_submit_button("Predict Score"):
            # Prepare input data
            input_data = {
                'Hours Studied': hours_studied,
                'Previous Scores': previous_score,
                'Extracurricular Activities': extracurricular,
                'Sleep Hours': sleep_hours,
                'Sample Question Papers Practiced': papers_solved
            }

            # Process and predict
            processed_data = preprocess_input(input_data, scaler, le)
            prediction = model.predict(processed_data)[0]
            
            # Display results
            st.success(f"Predicted performance score: {prediction:.2f}")

            # Prepare for storage
            input_data['prediction'] = float(prediction)
            input_data = {k: int(v) if isinstance(v, (int, np.integer)) else v 
                         for k, v in input_data.items()}
            
            # Store in MongoDB
            # collection.insert_one(input_data)

if __name__ == "__main__":
    main()
