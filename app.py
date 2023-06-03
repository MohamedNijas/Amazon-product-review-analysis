import streamlit as st
import pickle
import tensorflow as tf
import spacy
import gensim.downloader as api
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = tf.keras.models.load_model("pickle/model.h5")

nlp = spacy.load("en_core_web_lg")

tokenizer_file = "pickle/token.pkl"
with open(tokenizer_file, "rb") as f:
    tokenizer = pickle.load(f)

def word_preprocessing_vectorize(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_punct or token.is_stop:
            continue
        tokens.append(token.lemma_)
    return tokens
    

st.title("Sentiment Analysis-Amazon product review")
text = st.text_input("Type your product review here")

submitted = st.button("submit")

if submitted and text:
    with st.spinner("Processing..."):

        tokens = word_preprocessing_vectorize(text)
        sequences_test = tokenizer.texts_to_sequences(tokens)
        processed_text = pad_sequences(sequences_test,maxlen = 1000)
        prediction = model.predict(processed_text)[0][0]
        
        sentiment = "positive" if prediction > 0.5 else "negative"

    st.write("Sentiment:", sentiment)
