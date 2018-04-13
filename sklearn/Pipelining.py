from sklearn import linear_model ,decomposition ,datasets
from sklearn .pipeline import  Pipeline
from sklearn .model_selection  import  GridSearchCV
import matplotlib .pyplot as plt
import numpy as np

logistic = linear_model .LogisticRegression ()
pca = decomposition .PCA()

pipe = Pipeline (steps= [('pca',pca),('logistic',logistic )])

digits = datasets .load_digits ()
X_digits = digits .data
y_digits = digits .target

pca.fit(X_digits )
plt.figure(1,figsize= (4,3))
plt.clf()
plt.axes([0.2,0.2,0.7,0.7])
#axes([x,y,xs,ys])#其中x代表在X轴的位置，y代表在Y轴的位置，xs代表在X轴上向右延展的范围大小，yx代表在Y轴中向上延展的范围大小。
plt.plot (pca.explained_variance_ ,linewidth =2)
'''第一个是explained_variance_，它代表降维后的各主成分的方差值。方差值越大，则说明越是重要的主成分。
第二个是explained_variance_ratio_，它代表降维后的各主成分的方差值占总方差值的比例，
这个比例越大，则越是重要的主成分。

'''
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

n_components=[20,40,64]
Cs = np.logspace(-4,4,3)

estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, logistic__C=Cs))
estimator .fit(X_digits ,y_digits )
plt.axvline (estimator .best_estimator_ .named_steps['pca'].n_components,linestyle =':',label ='n_components chosen')
plt.legend(prop =dict(size=6))#图例的大小
plt.show()