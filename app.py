import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
s = PorterStemmer()



def text_transform(text):
    tex = text.lower()
    tex = nltk.word_tokenize(tex)
    y = []
    for i in tex:
        if i.isalnum() == True:
            y.append(i)

    tex = y[:]
    y.clear()

    for i in tex:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    tex = y[:]
    y.clear()

    for i in tex:
        y.append(s.stem(i))

    return " ".join(y)






tf = pickle.load(open('vectorizer.pkl','rb'))
model_mul = pickle.load(open('Model1.pkl','rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area('Enter the message')


if st.button('Predict'):
    # 1 processing
    transform_sms = text_transform(input_sms)

    # 2 vectorize
    vector_input = tf.transform([transform_sms])

    # 3 predict
    result = model_mul.predict(vector_input)[0]

    # 4 Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")