import numpy as np
x1= np.random.normal(size= 100)
x2= np.random .normal(size= 100)
x3=x1+x2
X= np.c_ [x1,x2,x3]
from sklearn import decomposition
pca = decomposition .PCA(n_components= 2)
#pca.fit(X)
X_reduced = pca.fit_transform(X )
print(X_reduced .shape)