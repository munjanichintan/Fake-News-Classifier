# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:57:01 2020

@author: Chintan Munjani
"""


from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
 	# df= pd.read_csv("spam.csv", encoding="latin-1")
 	# df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
 	# # Features and Labels
 	# df['label'] = df['class'].map({'ham': 0, 'spam': 1})
 	# X = df['message']
 	# y = df['label']
    
    
  #   # Extract Feature With CountVectorizer
 	# cv = CountVectorizer()
 	# X = cv.fit_transform(X)
 	# from sklearn.model_selection import train_test_split
 	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
 	# #Naive Bayes Classifier
 	

 	# clf = MultinomialNB()
 	# clf.fit(X_train,y_train)
 	# clf.score(X_test,y_test)
	with open('transform', 'rb') as f:
		cv = pickle.load(f)
	with open('nlp_model', 'rb') as file:
		pac = pickle.load(file)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = pac.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=False)