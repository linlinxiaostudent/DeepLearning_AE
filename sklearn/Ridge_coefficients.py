import matplotlib.pyplot  as plt
import numpy as np
from sklearn import linear_model
z=np.arange(1,11)
print(z)
z=np.arange(0,10)[:,np.newaxis ]#增加轴
print(z)
z=np.arange(1,11)+np.arange(0,10)[:,np.newaxis ]
print(z)
x = 1./(z )
y =  np.ones(10)
print(x)
print(y)

n_alpha= 200
alphas = np.logspace(-10,-2,n_alpha )
print(alphas )
coefs = []
for a in alphas :
    ridge = linear_model .Ridge(alpha= a,fit_intercept= False )
    ridge.fit(x,y)
    coefs.append(ridge.coef_)
ax = plt.gca()#显示当前的子图。可以利用plt.sca()进行subplot切换
ax.plot(alphas ,coefs )
ax.set_xscale('log')#把x轴设置为对数坐标
ax.set_xlim(ax.get_xlim()[::-1])#设置x的上下限
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')#显示坐标尺寸，:设置坐标轴的范围为数据的范围
plt.show()