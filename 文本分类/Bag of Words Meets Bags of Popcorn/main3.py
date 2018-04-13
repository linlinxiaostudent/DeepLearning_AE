
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import re
import nltk
#nltk.download('punkt')
from nltk.corpus import stopwords
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


data = pd.read_csv('labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)

def review_to_wordlist(raw_review,remove_stopwods=False ):
    # 这个函数的功能就是将原始的数据经过预处理变成一系列的词。
    # 输入是原始的数据（一条电影评论）。
    # 输出是一系列的词（经过预处理的评论）
    review_text = BeautifulSoup(raw_review,'lxml').get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text )
    words = letters_only.lower().split()
    if remove_stopwods :
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words

def review_to_sentences(review,tokenizer,remove_stopwords = False):
    raw_sentences =  tokenizer.tokenize(review.strip())
    sentence = []
    for raw_sentence in raw_sentences :
        if len(raw_sentence)>0:
            sentence.append(review_to_wordlist(raw_sentence ,remove_stopwords) )
    return sentence

num_reviews = data ['review'].size
clean_data_reviews = []
for i in range(0, num_reviews):
    if ((i + 1) % 1000 == 0):
        print('Review %d of %d \n' % (i + 1, num_reviews))
    clean_data_reviews.append(review_to_wordlist(data['review'][i]))

sentence = []
for i ,review in enumerate (data ['review']):
    sentence = sentence + review_to_sentences(review ,tokenizer )

print(len(clean_data_reviews))
print(len(sentence ))

# 恢复成训练集和测试集部分
train_x = clean_data_reviews[:20000]
label_train=data['sentiment'][:20000]
test_x = clean_data_reviews[20000:]
label_test=data['sentiment'][20000:]
