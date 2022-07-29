from flask import Flask, redirect, url_for, render_template, request
import joblib
import pandas as pd
import numpy as np
import nltk
nltk.data.path.append('./nltk_data/')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import pickle
import Sastrawi
import PyPDF2
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

multilabel_binarizer = pickle.load(open("multilabel_binarizer.sav", 'rb'))
xtrain = pickle.load(open("xtrain.sav", 'rb'))
model_brknn1 = joblib.load('model_brknn.pkl')
model_mlknn1 = joblib.load('model_mlknn.pkl')
model_br_svm1 = joblib.load('model_br_svm.pkl')
model_br_ova1 = joblib.load('model_br_ova.pkl')


def clean_text(text):
    text = re.sub("\'", "", text)
    text = re.sub("[^a-zA-Z]"," ",text)
    text = ' '.join(text.split())
    text = text.lower()
    return text

def stopword_eng(text):
    stop_words = set(stopwords.words('english'))
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

def stopword_ina(text):
    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
    stopword = factory.create_stop_word_remover()
    stop = stopword.remove(text)
    return stop
'''
def stemming_ina(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    katadasar = stemmer.stem(text)
    return(katadasar)
'''

def pre_process(text):
    text = clean_text(text)
    text = stopword_eng(text)
    text = stopword_ina(text)
    #text = stemming_ina(text)
    return text


def model_brknn(q):    
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features = 50)
    xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = model_brknn1.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)

def model_mlknn(q):    
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features = 50)
    xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = model_mlknn1.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)

def model_br_svm(q):    
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features = 50)
    xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = model_br_svm1.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)

def model_br_ova(q):    
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features = 50)
    xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = model_br_ova1.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)



@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def predict():
    pdffile = request.files["pdffile"]
    pdf_document1 = PyPDF2.PdfFileReader(pdffile)
    f_page1 = pdf_document1.getPage(0)
    text1 = f_page1.extractText()
    text2 = pre_process(text1)
    result1 = model_brknn(text2)
    result2 = model_mlknn(text2)
    result3 = model_br_svm(text2)
    result4 = model_br_ova(text2)
    return render_template('index.html', 
                            prediction1 = result1, 
                            prediction2 = result2, 
                            prediction3 = result3, 
                            prediction4 = result4,
                            textpdf = text2)

if __name__ == '__main__':
   app.run(port=3000)