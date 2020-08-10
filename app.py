# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 18:27:07 2020

@author: Chintan Munjani
"""

from flask import Flask,render_template,url_for,request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # # Read the data
    # df = pd.read_csv('news.csv')
    # df.drop(['Unnamed: 0', 'title'], axis=1, inplace=True)
    # df['label'] = df['label'].map({'FAKE': 1, 'REAL': 0})
    # labels = df.label
    
    # x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)
    
    # tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
    
    # tfidf_train=tfidf_vectorizer.fit_transform(x_train)
    
    # pac=PassiveAggressiveClassifier(max_iter=50)
    # pac.fit(tfidf_train,y_train)
    
    # with open('transform', 'wb') as f:
    #     pickle.dump(tfidf_vectorizer, f)
    # with open('nlp_model', 'wb') as file:
    #     pickle.dump(pac, file)
    
    with open('transform', 'rb') as f:
        tfidf = pickle.load(f)
    with open('nlp_model', 'rb') as file:
        pac1 = pickle.load(file)
        
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tfidf.transform(data).toarray()
        my_prediction = pac1.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=False)

    