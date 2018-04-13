import  numpy as np
import pandas as pd
import tensorflow as tf
import math
import os
checkpoints_dir = "./checkpoints"
#import pyplb as p
import tensorflow as tf

data=pd.read_csv('train.csv')#读入csv文件返回一个DataFrame对象
data['Sex']=data['Sex'].apply(lambda s:1 if s=='male' else 0)
mean_age=data['Age'].mean()
data .loc[data['Age'].isnull()]=mean_age
#data['Age'][data['Age'].isnull()]=mean_age

data=data.fillna(0)
dataset_X=data[['Sex','Age','Pclass','SibSp','Parch','Fare']]
dataset_X=dataset_X.as_matrix()
data['Deceased']=data['Survived'].apply(lambda s:int(not s))
dataset_Y=data[['Deceased','Survived']]
dataset_Y=dataset_Y.as_matrix()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataset_X,dataset_Y,test_size=0.2,random_state=42)

x=tf.placeholder(tf.float32 ,shape= [None,6])
y=tf.placeholder(tf.float32 ,shape= [None,2])

W=tf.Variable(tf.random_normal([6,2],name='weights'))
b=tf.Variable(tf.zeros([2]),name='bias')

y_pred=tf.nn.softmax(tf.matmul(x,W )+b)
cross_entropy=-tf.reduce_mean(y* tf.log(tf.maximum(0.00001, y_pred )) + (1.0 - y) * tf.log(tf.maximum(0.00001, 1.0 - y_pred )))
cost=tf.reduce_mean(cross_entropy)
train_op=tf.train.GradientDescentOptimizer(0.001).minimize(cost)
saver=tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(10):
        total_loss=0.
        for i in range(len(X_train)):
            feed={x:[X_train[i]],y:[y_train[i]]}
            _,loss=sess.run([train_op,cost],feed_dict=feed)
            total_loss+=loss
        print('Epoch: %04d,total_loss= %.9f' % (epoch+1,total_loss))
    print('Training complete')
    save_path = saver.save(sess, checkpoints_dir +"/trained_model.ckpt")
    print ("Model saved in file: %s" % save_path)

testdata = pd.read_csv('test.csv')
testdata['Sex']=testdata['Sex'].apply(lambda s:1 if s=='male' else 0)

mean_age=testdata['Age'].mean()
#testdata['Age'][testdata['Age'].isnull()]=mean_age
testdata .loc[testdata['Age'].isnull()]=mean_age
testdata=testdata.fillna(0)

X_test=testdata[['Sex','Age','Pclass','SibSp','Parch','Fare']]

with tf.Session() as sess1:
    saver.restore(sess1,save_path)
    predictions=np.argmax(sess1.run(y_pred,feed_dict={x:X_test }),1)

submission=pd.DataFrame({'PassengerId':testdata['PassengerId'],'Survived':predictions})
submission.to_csv('titanic-submission.csv',index=False)
print('End')

