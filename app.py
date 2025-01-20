import streamlit as st
from transformers import pipeline

st.title("Sentiment Classifier")

# Load the sentiment classification pipeline
classifier = pipeline('text-classification', model='./sentiment_transformer_model')  # Ensure path/model name is correct

# User Input
text = st.text_area("Enter Your Text Here")

# Predict Button
if st.button("Predict"):
    if text.strip():
        result = classifier(text)
        sentiment = result[0]['label']  # Extracting sentiment label
        confidence = result[0]['score']  # Extracting confidence score
        st.write(f"**Prediction:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.write("⚠️ Please enter some text.")
