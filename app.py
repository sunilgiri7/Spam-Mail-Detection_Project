import streamlit as st 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import sklearn

with open('spam-ham.pkl', 'rb') as f:
    model = pickle.load(f)

with open('cv.pkl', 'rb') as f:
    cv = pickle.load(f)

"""
Created on sat Aug 15    12:53:04 2022
@author: Sunil.Giri
"""
st.title(':red[Spam] or :blue[ham] mail :green[prediction] :sunglasses:')
qsn = st.text_input('Enter any text')

if st.button('PREDICT'):
    vectorized = cv.transform([qsn])
    result = model.predict(vectorized)

    if result:
        st.header('SPAM')
    else:
        st.header('HAM')