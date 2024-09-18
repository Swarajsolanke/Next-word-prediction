import tensorflow as tf
import nltk
nltk.download('gutenberg') # in this we get the text data 
from nltk.corpus import gutenberg
import pandas as pd 
import streamlit as st

#data preprocessing 
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import sklearn
from tensorflow.keras.models import load_model
import pickle 

#load the LSTM model 
model=load_model('D:\krishnaik course projects/Next word prediction/word_prediction.h5')
#load the tokenizer
with open('D:\krishnaik course projects/Next word prediction/tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

#function for predicting the next word 
def predict_next_words(model,tokenizer,text,max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_sequence_len:
        token_list=token_list[-(max_sequence_len-1):] # ensure the sequnece length matches max sequence length 
    token_list=pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicted=model.predict(token_list,verbose=0)
    predicted_word_index=np.argmax(predicted,axis=1)
    for word , index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None 

#streamlit app creation
st.title("Next Word Prediction based on previous context")
input_text=st.text_input("Enter the sentence")
if st.button('predict next word'):
    max_sequence_len=model.input_shape[1]+1
    next_word=predict_next_words(model,tokenizer,input_text,max_sequence_len)
    st.write(f'next word in the sequence is:{next_word}')
    # print(f'next word in the sequence is:{next_word}')








