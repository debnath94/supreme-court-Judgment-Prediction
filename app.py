# -*- coding: utf-8 -*-
"""
Created on Fri May 19 13:00:00 2023

@author: debna
"""

import streamlit as st
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the trained model
model = pickle.load(open('court_judgement.pickle', 'rb'))

vectorizer = pickle.load(open('vectorizer_court.pickle', 'rb'))


# Define the Streamlit app
def main():
    st.title("Supreme Court Judgment Prediction")
    st.write("Enter the case description:")
    
    # User input for case description
    case_description = st.text_area("")
    
    if st.button("Predict"):
        # Preprocess the case description (similar to training data preprocessing)
        preprocessed_case = preprocess(case_description)

        # Apply feature extraction (e.g., TF-IDF) on the preprocessed case
        transformed_case = vectorizer.transform([preprocessed_case])

        # Make prediction using the loaded model
        prediction = model.predict(transformed_case)[0]

        # Decode the prediction label if necessary
        # decoded_prediction = label_encoder.inverse_transform(prediction)

        # Display the prediction result
        st.write("Prediction:", prediction)
        if prediction == 1:
            st.success("1st party winning chances high😊!")
        else:
            st.error("2nd party winning chances low😥!")

#stop_words = set(stopwords.words('english'))
# Preprocessing function for the case description

def preprocess(case_description):
    # Lowercase the text
    preprocessed_text = case_description.lower()

    # Return the preprocessed text
    return preprocessed_text

if __name__ == '__main__':
    main()

















