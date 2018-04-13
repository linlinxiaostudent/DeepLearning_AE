import tensorflow as tf
import os


def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)#从第二行开始读
    key, value = reader.read(file_queue)
    #print(key,values)
    defaults = [[0], [0.], [0.], [0.], [0.], ['']]
    Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species = tf.decode_csv(value, defaults)
    #数据处理 
    preprocess_op = tf.case({
        tf.equal(Species, tf.constant('Iris-setosa')): lambda: tf.constant(0),
        tf.equal(Species, tf.constant('Iris-versicolor')): lambda: tf.constant(1),
        tf.equal(Species, tf.constant('Iris-virginica')): lambda: tf.constant(2),
    }, lambda: tf.constant(-1), exclusive=True)
    #栈
    return tf.stack([SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]), preprocess_op

def create_pipeline(filename, batch_size, num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    example, label = read_data(file_queue)
    #tf.train.string_input_producer：创建一个线程
    min_after_dequeue = 1000
    capacity = min_after_dequeue + batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )#tf.train.shuffle_batch:对队列中的样本进行乱序处理，返回的是从队列中随机筛选多个样例返回给image_batch，label_batch，

    return example_batch, label_batch

x_train_batch, y_train_batch = create_pipeline('Iris-train.csv', 50, num_epochs=1000)
x_test, y_test = create_pipeline('Iris-test.csv', 60)
global_step = tf.Variable(0, trainable=False)
learning_rate = 0.1#tf.train.exponential_decay(0.1, global_step, 100, 0.0)


# Input layer
x = tf.placeholder(tf.float32, [None, 4])
y = tf.placeholder(tf.int32, [None])

# Output layer
w = tf.Variable(tf.random_normal([4, 3]))
b = tf.Variable(tf.random_normal([3]))
a = tf.matmul(x, w) + b
prediction = tf.nn.softmax(a)

# Training
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=a, labels=y))
tf.summary.scalar('Cross_Entropy', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.cast(y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('Accuracy', accuracy)

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
merged_summary = tf.summary.merge_all()

sess = tf.Session()
train_writer = tf.summary.FileWriter('logs/train', sess.graph)
test_writer = tf.summary.FileWriter('logs/test', sess.graph)
sess.run(init)

coord = tf.train.Coordinator()#线程的协调器
threads = tf.train.start_queue_runners(sess=sess, coord=coord)#启动输入管道的线程，填充样本到队列中以便出队操作可以从队列中拿到样本。

try:
    print("Training: ")
    count = 0
    #curr_x_test_batch, curr_y_test_batch = sess.run([x_test, y_test])
    while not coord.should_stop():
        # Run training steps or whatever
        curr_x_train_batch, curr_y_train_batch = sess.run([x_train_batch, y_train_batch])

        sess.run(train_step, feed_dict={
            x: curr_x_train_batch,
            y: curr_y_train_batch
        })

        count += 1
        ce, summary = sess.run([cross_entropy, merged_summary], feed_dict={
            x: curr_x_train_batch,
            y: curr_y_train_batch
        })

        
        train_writer.add_summary(summary, count)
        curr_x_test_batch, curr_y_test_batch = sess.run([x_test, y_test])
        ce, test_acc, test_summary = sess.run([cross_entropy, accuracy, merged_summary], feed_dict={
            x: curr_x_test_batch,
            y: curr_y_test_batch
        })
        test_writer.add_summary(summary, count)
        print('Batch', count, 'J = ', ce,'测试准确率=',test_acc)
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop() #通知其他线程关闭

# Wait for threads to finish.
coord.join(threads)#其他所有线程关闭后，此函数返回
sess.close()
