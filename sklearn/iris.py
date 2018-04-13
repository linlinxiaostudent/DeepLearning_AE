from sklearn import datasets
from sklearn.svm import SVC
iris = datasets.load_iris()
print(len(iris .data))
clf =SVC()
clf.fit(iris.data,iris.target)
print(list(clf.predict(iris.data[:3])))
clf.fit(iris.data,iris.target_names[iris.target])
print(list(clf.predict(iris.data[:3])))