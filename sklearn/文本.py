from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# 选取参与分析的文本类别
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']
# 从硬盘获取原始数据
twenty_train=load_files('/mnt/vol0/20news-bydate-train',
        categories=categories,
        load_content = True,
        encoding='latin1',
        decode_error='strict',
        shuffle=True, random_state=42)
# 统计词语出现次数
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
# 使用tf-idf方法提取文本特征
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# 打印特征矩阵规格
print(X_train_tfidf.shape)

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# 选取参与分析的文本类别
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']
# 从硬盘获取原始数据
twenty_train=load_files('/mnt/vol0/20news-bydate-train',
        categories=categories,
        load_content = True,
        encoding='latin1',
        decode_error='strict',
        shuffle=True, random_state=42)
# 统计词语出现次数
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
# 使用tf-idf方法提取文本特征
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# 打印特征矩阵规格
print(X_train_tfidf.shape)

# 预测用的新字符串，你可以将其替换为任意英文句子
docs_new = ['Nvidia is awesome!']
# 字符串处理
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# 进行预测
predicted = clf.predict(X_new_tfidf)

# 打印预测结果
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))