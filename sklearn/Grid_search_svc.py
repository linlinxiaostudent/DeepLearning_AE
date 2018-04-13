import numpy as np
from sklearn .model_selection  import  GridSearchCV
from sklearn .model_selection import cross_val_score
from sklearn import datasets, svm
Cs= np.logspace(-6,-1,10)
svc = svm.SVC(kernel='linear')

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

clf = GridSearchCV (estimator= svc ,param_grid= dict (C=Cs))
clf.fit(X_digits[:1000], y_digits[:1000])
score_best = clf.best_score_
print(score_best )
C_best = clf.best_estimator_ .C

score_train = clf.score(X_digits[1000:], y_digits[1000:])
print(score_train )