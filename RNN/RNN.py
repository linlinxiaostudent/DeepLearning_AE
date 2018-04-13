import  random
import numpy as np

import  tensorflow  as tf
from  tensorflow.contrib.rnn.python.ops import core_rnn
from  tensorflow.contrib.rnn.python.ops import core_rnn_cell
sess=tf.InteractiveSession()
length=10
time_step_size=length
vector_size =1
batch_size=10
test_size=10

def bulid_data(n):
    xs=[]
    ys=[]
    for i in range(2000):
        k = random.uniform(1,50)
        x = [[np.sin(k+j)] for j in range(0,n)]
        y = [np.sin(k+n)]
        xs.append(x)
        ys.append(y)
    train_x=np.array(xs[0:1500])
    train_y = np.array(ys[0:1500])
    test_x = np.array(xs[1500:])
    test_y = np.array(ys[1500:])
    return (train_x,train_y,test_x,test_y)

(train_x,train_y,test_x,test_y)=bulid_data(length)
print(train_x. shape ,train_y .shape ,test_x .shape ,test_y .shape )

X=tf.placeholder('float',[None,length,vector_size ])
Y=tf.placeholder('float',[None ,1])

W=tf.Variable (tf.random_normal([10,1],stddev= 0.01) )
B=tf.Variable (tf.random_normal([1],stddev= 0.01))

tf.summary.histogram("W",W)
tf.summary.histogram("B",B)

def seq_pridect_models(X,w,b,time_step_size,vector_size):
    X=tf.transpose(X,[1,0,2])
    X=tf.reshape(X,[-1,vector_size])
    X=tf.split(X,time_step_size ,0)
    cell=core_rnn_cell.BasicRNNCell (num_units= 10)
    initial_state=tf.zeros([batch_size ,cell.state_size ])
    outputs,_state =core_rnn.static_rnn(cell,X,initial_state= initial_state )
    return  tf.matmul(outputs [-1],w)+b,cell.state_size

def seq_pridect_model(X,w,b,time_step_size,vector_size):
    X=tf.transpose(X,[1,0,2])
    X=tf.reshape(X,[-1,vector_size])
    X=tf.split(X,time_step_size ,0)
    cell=core_rnn_cell.BasicLSTMCell  (num_units= 10,forget_bias= 1.0,state_is_tuple= True )
    outputs,_state =core_rnn.static_rnn(cell,X,dtype= tf.float32  )
    return  tf.matmul(outputs [-1],w)+b,cell.state_size

pred_y,_=seq_pridect_model(X,W,B,time_step_size ,vector_size )
loss =tf.square(tf.subtract(Y,pred_y ) )
train_op=tf.train.GradientDescentOptimizer(0.001).minimize(loss )
init =tf.global_variables_initializer()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("",sess.graph)
sess.run(init)
for i in range(50):
    for end in range(batch_size ,len(train_x ),batch_size ):
        begin=end-batch_size
        x_value=train_x[begin :end ]
        y_value=train_y [begin :end]
        summary,_=sess.run([merged,train_op] ,feed_dict= {X:x_value ,Y:y_value })
    writer.add_summary(summary, i)
    test_indices=np.arange(len(test_x ))#生成等差数列
    np.random .shuffle(test_indices )
    test_indices =test_indices [0:test_size ]
    x_value =test_x [test_indices ]
    y_value =test_y [test_indices ]
    val_loss=np.mean(sess.run(loss,feed_dict= {X:x_value ,Y:y_value }) )
    print('Run %s'% i,val_loss )
