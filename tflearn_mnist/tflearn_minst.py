import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
learn = tf.contrib.learn
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = learn.datasets.load_dataset('mnist')
data = mnist.train.images
labels = np.asarray(mnist.train.labels, dtype=np.int32)
test_data = mnist.test.images
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)
max_examples = 10000
data = data[:max_examples]
labels = labels[:max_examples]

def display(i):
    img = test_data[i]
    plt.title('Example %d. Label: %d' % (i, test_labels[i]))
    plt.imshow(img.reshape((28, 28)), cmap=plt.cm.gray_r)
    plt.show()

def test_display():
    display(0)
    display(1)
    display(8)
    print(len(data[0]))

feature_columns = learn.infer_real_valued_columns_from_input(data)
classifier = learn.LinearClassifier(feature_columns=feature_columns, n_classes=10)
classifier.fit(data, labels, batch_size=100, steps=1000)
result = classifier.evaluate(test_data, test_labels)
print(result["accuracy"])

def input_fn_predict(): # returns x, None
    #do your check here or you can just print it out
    feature_tensor = tf.constant(test_labels[0],shape=[1,test_data[0].size])
    return feature_tensor,None

predictions=classifier.predict_classes(input_fn= input_fn_predict)
predictions=[i for i in predictions]
# here's one it gets right

print ("Predicted %d, Label: %d" % (int(predictions[0]),test_labels[0]))
#display(0)

print(classifier.get_variable_names())
weights = classifier.get_variable_value('linear//weight')
f, axes = plt.subplots(2, 5, figsize=(10, 4))
axes = axes.reshape(-1)

for i in range(len(axes)):
    a = axes[i]
    a.imshow(weights.T[i].reshape(28, 28), cmap=plt.cm.seismic)
    a.set_title(i)
    a.set_xticks(())  # ticks be gone
    a.set_yticks(())
plt.show()
