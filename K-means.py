import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
#随机生成数据，2000个点，随机分为两类
num_puntos=2000
conjunto_puntos=[]#数据存放在列表中
for i in range(num_puntos):
    if np.random.random()<0.5:
        conjunto_puntos.append([np.random.normal(0.0,0.9),np.random.normal(0.0,0.9)])
    else:
        conjunto_puntos.append([np.random.normal(3.0,0.5),np.random.normal(3.0,0.5)])

df=pd.DataFrame({'x':[v[0] for v in conjunto_puntos],
                 'y':[v[1] for v in conjunto_puntos]})
sns.lmplot('x','y',data=df,fit_reg=False,size=6)
plt.show()
#lmplot, 首先要明确的是:它的输入数据必须是一个Pandas的'DataFrame Like' 对象,
#然后从这个DataFrame中挑选一些参数进入绘图充当不同的身份.

#把数组装进tensor中
vectors=tf.constant(conjunto_puntos)
k=2
#tf.random_shuffle（value,seed=None,name=None）：对value（是一个tensor）的第一维进行随机化。相当于把数据随机化
centroides=tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))
#随机选取K个点作为质心

#增加维度
expanded_vectors=tf.expand_dims(vectors,0)
expanded_centroides=tf.expand_dims(centroides,1)

diff=tf.sub(expanded_vectors,expanded_centroides)
sqr=tf.square(diff)
distance=tf.reduce_sum(sqr,2)

#挑选每一个点里的最近的质心（返回的是最小值索引）
assignments=tf.argmin(distance,0)

#tf.where（）返回bool型tensor中为True的位置，
#tf.gather(params, indices, validate_indices=None, name=None)合并索引indices所指示params中的切片
means=tf.concat(0,[tf.reduce_mean(tf.gather(vectors,
                                            tf.reshape(tf.where(tf.equal(assignments,c)),[1,-1])),
                                  reduction_indices=[1])for c in range(k)])
#tf.assign()用means更新centroides
update_centroides=tf.assign(centroides,means)


init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for step in range(100):
    _,centroid_values,assignment_values=sess.run([update_centroides,centroides,assignments])
data={'x':[],'y':[],'cluster':[]}

for i in range(len(assignment_values)):
    data['x'].append(conjunto_puntos[i][0])
    data['y'].append(conjunto_puntos[i][1])
    data['cluster'].append(assignment_values[i])
df=pd.DataFrame(data)   
sns.lmplot('x','y',data=df,fit_reg=False,size=6,hue='cluster',legend=False)
#hue通过指定一个分组变量, 将原来的y~x关系划分成若干个分组;fit_reg：是否显示回归曲线
plt.show()















    



