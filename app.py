import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf
import numpy as np
import contractions
import re
from bs4 import BeautifulSoup
from string import punctuation
import nltk
nltk.download('punkt')
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import urllib.request
import os.path

st.set_page_config(layout='centered')

st.cache()
def download_doc2vec_vectores():
    url = 'https://github.com/mazy06000/negative-reviews-detection/releases/download/model/doc2vec_model.dv.vectors.npy'
    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, filename)

def to_stemming(text):
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
                'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
                "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
                'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                'with', 'about', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
                'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
                't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                "aren't", "couldn't", "didn't", "doesn't", "hadn't", "hasn't", "haven't", "isn't", 'ma', "mightn't", "mustn't", "needn't", "shan't",
                "shouldn't", "wasn't", "weren't", "won't", "wouldn't"]
    
    # Expanding
    expanded_words = []    
    for word in text.split():
      # using contractions.fix to expand the shotened words
      expanded_words.append(contractions.fix(word))   
    text = ' '.join(expanded_words)
    
    text = text.lower() # lower
    text = re.sub(pattern=r'https?://\S+|www\.\S+', repl='', string=text) # remove urls
    text = BeautifulSoup(text, 'lxml').get_text() # remove html content
    text = re.sub(r'@[A-Za-z0-9]+', repl='', string=text) # remove user name @
    text = text.translate(str.maketrans('', '', punctuation)) # ponctuation

    
    # tokenization
    text_final = []
    text = word_tokenize(text)

    # stemming
    stemmer = PorterStemmer()
    for index in range(len(text)):
        # stem word to each word
        stem_word = stemmer.stem(text[index])
        # update tokens list with stem word
        text_final.append(stem_word)
        
    # removing stopwords and joining all tokens
    text_without_sw = [w for w in text_final if w not in stop_words]
    text = ' '.join(text_without_sw)
        
    return text

if "model" not in st.session_state:
    if not os.path.exists("doc2vec_model.dv.vectors.npy"):
        download_doc2vec_vectores()
    st.session_state['doc_model'] = Doc2Vec.load("doc2vec_model")
    st.session_state['model'] = tf.keras.models.load_model("negative_reviews.h5")

def predict(text):
    text_stemmed = to_stemming(text)
    text_tokenized = word_tokenize(text_stemmed)
    tagged_document = [TaggedDocument(doc, [i]) for i, doc in enumerate([text_tokenized])]
    text_vectorized = np.array([st.session_state['doc_model'].infer_vector(doc.words) for doc in tagged_document])

    return st.session_state['model'].predict(text_vectorized)[0][0]

def plot_indicator(percent, color, sentence, result):
    if percent is not None:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=percent,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence (%)"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': color},
                   'bordercolor': color},
        ))

        fig.update_layout(width=500, height=500)

        header.plotly_chart(fig, use_container_width=True)
        header.markdown(
            f'<div style="text-align: center; font-size: 30px; font-weight: bold; color:{color}">{result}</div>',
            unsafe_allow_html=True)
        header.markdown(
            f'<div style="text-align: center;">Text: {sentence}</div>',
            unsafe_allow_html=True)

    else:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=0,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence"},
            gauge={'axis': {'range': [None, 100]},
                   'bordercolor': color},
        ))

        fig.update_layout(width=500, height=500)

        header.plotly_chart(fig, use_container_width=True)


header = st.container()
text = st.container()
characteristic = st.container()

with header:
    st.markdown("""<div style="display:flex;justify-content:center;"><h1>BAD BUZZ DETECTION</h1></div>""", unsafe_allow_html=True)

with text:
    sentence = st.text_input(label="Sentence to predict if is a bad buzz or not",
                             placeholder="Your sentence here")

    if sentence:

        y_pred_proba = predict(sentence)
        y_pred = (y_pred_proba > 0.5) + 0

        if y_pred == 1:
            result = "BAD BUZZ"
            percent = round(y_pred_proba * 100)
            plot_indicator(percent=percent, color='red', sentence=sentence, result=result)
        else:
            result = "NOT BAD BUZZ"
            percent = round((1 - y_pred_proba) * 100)
            plot_indicator(percent=percent, color='green', sentence=sentence, result=result)
    else:
        plot_indicator(percent=None, color='green', sentence=None, result=None)

with characteristic:
    st.header("Characteristic")
    left, right = st.columns(2)
    left.subheader("About Model")
    left.markdown('<div><b>Model:</b> Basic Neural Network</div>',
                  unsafe_allow_html=True)
    left.markdown('<div><b>Word Embedding:</b> Doc2Vec</div>',
                  unsafe_allow_html=True)

    right.subheader("Model performance")
    right.markdown('<div><b>Test AUC:</b> 78%</div>', unsafe_allow_html=True)
    right.markdown('<div><b>Test Accuracy:</b> 70%</div>', unsafe_allow_html=True)