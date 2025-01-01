import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
nltk.download('punkt')  
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize

tfidf=pickle.load(open('Vectorizer.pkl','rb'))
model=pickle.load(open('mnb.pkl','rb'))


nltk.download('stopwords')
def transform_text(text):
    
    # convert the text into lower case
    text=text.lower()
    
    # separate the words and make the list
    text=nltk.word_tokenize(text)
    
    # remove the special character
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    # remove the stop words and punctuation
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # perform stemming like loving,loved,love will be converted to love
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    # return the text as a string
    return " ".join(y)

st.title("Email/SMS Classifier")

input_sms=st.text_area("Enter the Email: ")

# Data PreProcessing

if st.button("Predict"):
    transformed_sms=transform_text(input_sms)

    #vectorize
    vector_sms=tfidf.transform([transformed_sms]).toarray()


    # predict
    result=model.predict(vector_sms)
    if result==1:
        st.header("The Email is a Spam")
    else:
        st.header("The Email is not a spam")
