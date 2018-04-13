import numpy as np
from sklearn.model_selection  import  cross_val_score
from sklearn import datasets, svm

digits = datasets.load_digits()
X = digits.data
y = digits.target

svc = svm.SVC(kernel='linear')
C_s = np.logspace(-10, 0, 10)

scores = list()
scores_std = list()

for C in C_s:
    svc.C = C
    this_scores = cross_val_score(svc, X, y, n_jobs=1)#
    #print(this_scores)
    #print(np.mean(this_scores ))
    #print(np.std(this_scores ) )
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))

# 绘制图象
import matplotlib.pyplot as plt
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.semilogx(C_s, scores)#设置X坐标为对数坐标
plt.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--')
plt.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--')
locs, labels = plt.yticks()#设置轴标记
print(locs )
print(labels )
plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
#‘%f’和‘%e’默认情况下都会保留小数点后面六位有效数字，‘%g’在保证六位有效数字的前提下，使用小数方式，否则使用科学计数法。
plt.ylabel('CV score')
plt.xlabel('Parameter C')
plt.ylim(0, 1.1)#Y轴范围
plt.show()


