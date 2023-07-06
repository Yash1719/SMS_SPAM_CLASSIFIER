import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

tf = pickle.load(open('vectorizer(1).pkl', 'rb'))
model = pickle.load(open('model(1).pkl', 'rb'))

st.title('Email/sms spam classifier')

input_sms = st.text_area("Enter the message")

def text_transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


if st.button('Predict'):
    transformed_text = text_transform(input_sms)
    vector_input = tf.transform([transformed_text])
    result = model.predict(vector_input)[0]

    if result == 0:
        st.header("Not Spam")
    else:
        st.header("SPAM")
