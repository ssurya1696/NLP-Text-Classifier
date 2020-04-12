
# coding: utf-8

# In[1]:


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
import pickle


# In[2]:


data = pd.read_csv('data.csv',engine='python')


# In[3]:


data = shuffle(data)


# In[4]:


dict = {"Green": 0, "Yellow": 1, "Red": 2}


# In[5]:


data = data.replace({"code": dict})


# In[6]:


stopwords = set(stopwords.words('english')) 


# In[7]:


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


# In[8]:


preprocessed_text = []
# tqdm is for printing the status bar
for sentance in tqdm(data['text'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    
    sent = ' '.join(e for e in sent.split() if e not in stopwords)
    preprocessed_text.append(sent.lower().strip())
data['preprocessed_text'] = preprocessed_text


# In[9]:


neg = []
pos = []
neu = []
compound = []

sid = SentimentIntensityAnalyzer()


# In[10]:


for for_sentiment  in tqdm(data['preprocessed_text']):

    neg.append(sid.polarity_scores(for_sentiment)['neg']) #Negative Sentiment score
    pos.append(sid.polarity_scores(for_sentiment)['pos']) #Positive Sentiment score
    neu.append(sid.polarity_scores(for_sentiment)['neu']) #Neutral Sentiment score
    compound.append(sid.polarity_scores(for_sentiment)['compound']) #Compound Sentiment score

# Creating new features    
data['neg_ss']      = neg
data['pos_ss']      = pos
data['neu_ss']      = neu
data['compound_ss'] = compound


# In[11]:


Y = data['code'].values
data.drop(['text'], axis=1, inplace=True)


# In[12]:


data.drop(['code'], axis=1, inplace=True)


# In[13]:


X = data


# In[14]:


vectorizer = TfidfVectorizer()
vectorizer.fit(X['preprocessed_text'])


# In[15]:


pickle.dump(vectorizer, open('tfidf_tranform.pkl', 'wb'))


# In[16]:




#tidf Train Data
text_tfidf_train = vectorizer.transform(X['preprocessed_text'])


# In[18]:


normalizer = Normalizer()
normalizer.fit(X['neg_ss'].values.reshape(-1,1))

pickle.dump(normalizer, open('neg_ss_tranform.pkl', 'wb'))


# In[19]:


neg_ss_data_train = normalizer.transform(X['neg_ss'].values.reshape(-1,1))


# In[20]:


normalizer = Normalizer()
normalizer.fit(X['pos_ss'].values.reshape(-1,1))

pickle.dump(normalizer, open('pos_ss_tranform.pkl', 'wb'))


# In[26]:




pos_ss_data_train = normalizer.transform(X['pos_ss'].values.reshape(-1,1))


# In[21]:


normalizer = Normalizer()
normalizer.fit(X['neu_ss'].values.reshape(-1,1))

pickle.dump(normalizer, open('neu_ss_tranform.pkl', 'wb'))


# In[27]:




neu_ss_data_train = normalizer.transform(X['neu_ss'].values.reshape(-1,1))


# In[22]:


normalizer = Normalizer()
normalizer.fit(X['compound_ss'].values.reshape(-1,1))

pickle.dump(normalizer, open('compound_ss_tranform.pkl', 'wb'))


# In[23]:




compound_ss_data_train = normalizer.transform(X['compound_ss'].values.reshape(-1,1))


# In[28]:


X_train_merge = hstack((text_tfidf_train,neg_ss_data_train,pos_ss_data_train,neu_ss_data_train,compound_ss_data_train)).tocsr()


# In[29]:


x_cfl=XGBClassifier()

prams={
    'learning_rate':[0.01,0.03,0.05,0.1,0.15,0.2],
     'n_estimators':[100,200,500,1000,2000],
     'max_depth':[3,5,10],
    'colsample_bytree':[0.1,0.3,0.5,1],
    'subsample':[0.1,0.3,0.5,1]
}
random_cfl=RandomizedSearchCV(x_cfl,param_distributions=prams,verbose=10,n_jobs=-1,cv=5,scoring='balanced_accuracy')
random_cfl.fit(X_train_merge, Y)


# In[30]:


x_cfl=XGBClassifier()


# In[31]:


x_cfl.set_params(**random_cfl.best_params_)


# In[32]:


x_cfl.fit(X_train_merge,Y,verbose=True)


# In[33]:


# Saving model to disk
pickle.dump(x_cfl, open('model.pkl','wb'))

