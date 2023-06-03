#Amazon Product Review Sentiment Analysis
This is a simple application that performs sentiment analysis on Amazon product reviews. It uses a machine learning model to predict whether a given review is positive or negative.
You can access the deployed application by clicking this link: 
Installation
To run this application, you need to install the following dependencies:

streamlit
pickle
tensorflow
spacy
gensim
numpy
You can install them using the following command:

shell
Copy code
pip install streamlit pickle tensorflow spacy gensim numpy
Usage
Make sure you have the trained model and tokenizer files in the correct locations. The model file should be named "model.h5" and placed in the "pickle" folder. The tokenizer file should be named "token.pkl" and also placed in the "pickle" folder.

Run the following command to start the application:

shell
Copy code
streamlit run app.py
Once the application is running, you can enter a text in the input field labeled "Type here your Essay topic".

Click the "submit" button to process the text and perform sentiment analysis.

The application will display the sentiment of the input text as either "positive" or "negative".

Note: The sentiment analysis model used in this application is trained on Amazon product reviews, so it may not generalize well to other types of texts.

Additional Information
The sentiment analysis model is loaded from the "model.h5" file using TensorFlow's load_model function.

The English language model from spaCy ("en_core_web_lg") is loaded for text preprocessing.

The tokenizer used for converting text to sequences is loaded from the "token.pkl" file using the pickle library.

The input text is preprocessed by lemmatizing the tokens and removing punctuation and stop words.

The preprocessed text is then padded to a fixed length of 1000 tokens using TensorFlow's pad_sequences function.

The sentiment prediction is made using the loaded model, and the predicted sentiment is displayed on the screen.
