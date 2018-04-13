from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 下载并导入MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 参数
learning_rate = 0.01#学习率
training_epochs = 20#训练的周期
batch_size = 256#每一批次训练的大小
display_step = 1
examples_to_show = 10

# 神经网络的参数
n_input=784
n_hidden_1 = 256 # 隐层1的神经元个数
n_hidden_2 = 128 # 隐层2神经元个数
n_output = 10 # MNIST数据集的输出(img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])
Y=tf.placeholder('float',[None,10])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'softmax_w': tf.Variable(tf.random_normal([n_hidden_2, n_output])),
    
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'softmax_b': tf.Variable(tf.random_normal([n_output])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

y_pred=tf.nn.softmax(tf.matmul(encoder(X),weights['softmax_w'])+biases['softmax_b'])
# Define loss and optimizer, minimize the squared error

cost = tf.reduce_mean(tf.pow(Y - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)

total_batch = int(mnist.train.num_examples/batch_size)
# Training cycle
for epoch in range(training_epochs):
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs,Y:batch_ys})
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1),
              "cost=", "{:.9f}".format(c))

print("Optimization Finished!")

