from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from tqdm import tqdm
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.model_selection import RandomizedSearchCV

# load the model from disk
filename = 'model.pkl'
clf = pickle.load(open(filename, 'rb'))
tfidf=pickle.load(open('tfidf_tranform.pkl','rb'))
pos_ss=pickle.load(open('pos_ss_tranform.pkl','rb'))
neg_ss=pickle.load(open('neg_ss_tranform.pkl','rb'))
neu_ss=pickle.load(open('neu_ss_tranform.pkl','rb'))
compound_ss=pickle.load(open('compound_ss_tranform.pkl','rb'))
app = Flask(__name__)

stopwords = set(stopwords.words('english')) 

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	# clf = joblib.load(NB_spam_model)

	if request.method == 'POST':
		message = request.form['message']
		data = message
		data = {'text':[data]}
		data = pd.DataFrame(data)
		preprocessed_text = []
		for sentance in (data['text'].values):
			sent = decontracted(sentance)
			sent = sent.replace('\\r', ' ')
			sent = sent.replace('\\"', ' ')
			sent = sent.replace('\\n', ' ')
			sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    
			sent = ' '.join(e for e in sent.split() if e not in stopwords)
			preprocessed_text.append(sent.lower().strip())
		data['preprocessed_text'] = preprocessed_text
		neg = []
		pos = []
		neu = []
		compound = []

		sid = SentimentIntensityAnalyzer()
		
		for for_sentiment  in tqdm(data['preprocessed_text']):
			neg.append(sid.polarity_scores(for_sentiment)['neg']) #Negative Sentiment score
			pos.append(sid.polarity_scores(for_sentiment)['pos']) #Positive Sentiment score
			neu.append(sid.polarity_scores(for_sentiment)['neu']) #Neutral Sentiment score
			compound.append(sid.polarity_scores(for_sentiment)['compound']) #Compound Sentiment score
			
		data['neg_ss']      = neg
		data['pos_ss']      = pos
		data['neu_ss']      = neu
		data['compound_ss'] = compound	
		
		text_tfidf_test = tfidf.transform(data['preprocessed_text'])
		neg_ss_data_test = neg_ss.transform(data['neg_ss'].values.reshape(-1,1))
		pos_ss_data_test = pos_ss.transform(data['pos_ss'].values.reshape(-1,1))
		neu_ss_data_test = neu_ss.transform(data['neu_ss'].values.reshape(-1,1))
		compound_ss_data_test = compound_ss.transform(data['compound_ss'].values.reshape(-1,1))
		
		X_test_merge = hstack((text_tfidf_test,neg_ss_data_test,pos_ss_data_test,neu_ss_data_test,compound_ss_data_test)).tocsr()
		
		
		my_prediction = clf.predict(X_test_merge)
		
				
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)