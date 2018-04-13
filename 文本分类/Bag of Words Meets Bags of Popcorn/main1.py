
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF


data = pd.read_csv('labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)


def review_to_words(raw_review):
    # 这个函数的功能就是将原始的数据经过预处理变成一系列的词。
    # 输入是原始的数据（一条电影评论）。
    # 输出是一系列的词（经过预处理的评论）
    review_text = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return (" ".join(meaningful_words))

num_reviews = data ['review'].size
clean_data_reviews = []
for i in range(0, num_reviews):
    if ((i + 1) % 1000 == 0):
        print('Review %d of %d \n' % (i + 1, num_reviews))
    clean_data_reviews.append(review_to_words(data['review'][i]))


tfidf = TFIDF(min_df=2,max_features=None,
              strip_accents='unicode',analyzer='word',
              token_pattern=r'\w{1,}',ngram_range=(1, 3),
              use_idf=1,smooth_idf=1,
              sublinear_tf=1,stop_words = 'english') # 去掉英文停用词
# 合并训练和测试集以便进行TFIDF向量化操作
tfidf.fit(clean_data_reviews)
clean_data_reviews = tfidf.transform(clean_data_reviews)
# 恢复成训练集和测试集部分
train_x = clean_data_reviews[:20000]
label_train=data['sentiment'][:20000]
test_x = clean_data_reviews[20000:]
label_test=data['sentiment'][20000:]

from sklearn.naive_bayes import MultinomialNB as MNB

model_NB = MNB()
model_NB.fit(train_x, label_train)
MNB(alpha=1.0, class_prior=None, fit_prior=True)

from sklearn.cross_validation import cross_val_score
import numpy as np

print ("多项式贝叶斯分类器10折交叉验证得分: ",
       np.mean(cross_val_score(model_NB, train_x, label_train , cv=10, scoring='roc_auc')))


test_predicted = np.array(model_NB.predict(test_x))

label_test_array = []
for i in range(20000, num_reviews):
    label_test_array.append(label_test[i])
num =0
for i in range(0,len(label_test_array)):
    if (test_predicted[i]==label_test_array[i]):
        num=num+1

print('acc:',num/len(label_test ))