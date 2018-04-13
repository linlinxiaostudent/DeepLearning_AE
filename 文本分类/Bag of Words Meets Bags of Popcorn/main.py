
from keras.models import Sequential
from keras.layers import LSTM ,Embedding,Dense,Activation
from keras import backend as K
from bs4 import BeautifulSoup
import keras
import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

hidden_size = 10000
embedded_size = 5000
batch_size = 100
max_features=10000
data = pd.read_csv('labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
train=data[0:20000]
test=data[20000:len(data['review'])]

def review_to_words( raw_review ):
    # 这个函数的功能就是将原始的数据经过预处理变成一系列的词。
    # 输入是原始的数据（一条电影评论）。
    # 输出是一系列的词（经过预处理的评论）
    review_text = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return( " ".join( meaningful_words ))

num_reviews = train['review'].size
clean_train_reviews = []
for i in range(0, num_reviews):
    if ((i+1)%1000==0):
        print('Review %d of %d \n' % (i+1,num_reviews))
    clean_train_reviews.append(review_to_words(train['review'][i]))

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

num_reviews=len(data ['review'])
clean_test_reviews = []
for i in range(20000,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print ("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

model = Sequential()
model.add(Embedding(5000, 1000))

model.add(LSTM(64))
model.add(Dense(1))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data_features,train["sentiment"],batch_size=batch_size,epochs=20)
                
#评估一下模型的效果
model.evaluate(test_data_features, test["sentiment"],verbose=True, batch_size=batch_size)
