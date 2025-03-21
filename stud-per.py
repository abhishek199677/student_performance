import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Creating collection inside MongoDB database
uri = "mongodb+srv://email:password@cluster0.zfgws.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['student']
collection = db['student_prediction']


def load_model():
    with open("student_linearREG_final_model.pkl", 'rb') as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le


def preprocesssing_input_data(data, scaler, le):
    # Ensure the input value is properly encoded
    return scaler.transform(pd.DataFrame(data, index=[0]))


def predict_data(data):  # here predict data is calling the above two functions i.e load_model and preprocesssing_input_data
    model, scaler, le = load_model()
    processed_data = preprocesssing_input_data(data, scaler, le)
    prediction = model.predict(processed_data)
    return prediction


def main():
    st.markdown(
        """
    <h1 style='
        text-align: center;
        color: #808080; /* Ash color */
        font-family: Arial, sans-serif; /* Changed to Arial */
        text-shadow: 2px 2px 4px #000000;
        border-bottom: 2px solid #3366FF;
        padding-bottom: 10px;
    '>
    Students Performance Prediction
    </h1>
    """,
        unsafe_allow_html=True,
    )
    # st.title("Students Performance Prediction")
    # st.write("Enter your data to get a prediction for your performance")  # Removed redundant line

    st.markdown(
        """
    <div style='
        text-align: center;
        font-size: 2em;
        color: white;
        background-color: #4682B4; /* Steel Blue */
        font-family: "Times New Roman", serif;
        padding: 5px;
        border-radius: 5px;
        box-shadow: 5px 5px 10px #888888;
    '>
        Enter your data to get a prediction for your performance
    </div>
    """,
        unsafe_allow_html=True,
    )

    hour_sutdied = st.number_input("Hours studied", min_value=1, max_value=10, value=5)
    prvious_score = st.number_input("Previous score", min_value=40, max_value=100, value=70)
    extra = st.selectbox("Extra curricular activites", ['Yes', "No"])
    sleeping_hour = st.number_input("Sleeping hours", min_value=4, max_value=10, value=7)
    number_of_peper_solved = st.number_input("Number of question paper solved", min_value=0, max_value=10, value=5)

    # Load the model, scaler, and LabelEncoder
    model, scaler, le = load_model()  # Load them here!

    if st.button("Predict your score"):  # data mapping

        # Use the same LabelEncoder instance to transform the input
        extra_encoded = le.transform([extra])[0]

        user_data = {
            "Hours Studied": hour_sutdied,
            "Previous Scores": prvious_score,
            "Extracurricular Activities": extra_encoded,  # Use the *converted* value
            "Sleep Hours": sleeping_hour,
            "Sample Question Papers Practiced": number_of_peper_solved
        }

        prediction = predict_data(user_data)
        st.success(f"your prediciotn result is {prediction}")

        # Convert prediction to a standard Python float before storing
        user_data['prediction'] = round(float(prediction[0]), 2)  # Access the first element of the NumPy array
        user_data = {key: int(value) if isinstance(value, np.integer) else float(value) if isinstance(value,
                                                                                                        np.floating) else value
                     for key, value in user_data.items()}
        collection.insert_one(user_data)


if __name__ == "__main__":
    main()
