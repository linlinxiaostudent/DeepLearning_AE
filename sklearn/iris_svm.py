import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y != 0, :2]
y = y[y != 0]


# 样本容量
n_sample = len(X)
#print(n_sample )

# 产生随机序列用于划分训练集/测试集
np.random.seed(0)
order = np.random.permutation(n_sample)
#print(order )
X = X[order]
#print(X)
y = y[order].astype(np.float)
#print(y )
X_train = X[:int(.9 * n_sample)]
y_train = y[:int(.9 * n_sample)]
X_test = X[int(.9 * n_sample):]
y_test = y[int(.9 * n_sample):]

# 对不同kernel分别训练模型
for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(X_train, y_train)

    plt.figure(fig_num)#创建新的figure
    plt.clf()#清除画布
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)

    # 输出测试数据图表
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    print(XX)
    print(YY)
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])#样本点到超平面的距离。np.c_:将两个数据沿着第二个轴合并。

    # 将计算结果加入图表
    Z = Z.reshape(XX.shape)
    print(Z)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)#填充等高线的颜色
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])#绘制等高线

    plt.title(kernel)
plt.show()
