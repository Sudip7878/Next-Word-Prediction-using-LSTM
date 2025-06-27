# Step 1: Import Libraries and Load the Model
import numpy as np
#import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model
model = load_model('lstm_hamlet_model.h5')

# Load the Tokenizer
import pickle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len=14):
    # Tokenize the input text
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Keep only the last max_sequence_len-1 tokens  
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted= model.predict(token_list, verbose=0)
    predict_next_index = np.argmax(predicted, axis=1)

    for word, index in tokenizer.word_index.items():
        if index == predict_next_index:
            return word
    
    return None

# Streamlit app
import streamlit as st
st.title('Next Word Prediction with LSTM')
input_text = st.text_input('Enter a sentence:', 'To be or not to be')
if st.button('Predict Next Word'):
    max_sequence_len = 14
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Predicted next word: {next_word}')
