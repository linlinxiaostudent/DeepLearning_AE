import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics

digits= datasets.load_digits()
images_and_labels = list(zip(digits.images,digits.target))

for index,(image,label) in enumerate (images_and_labels[:4]):
    plt.subplot(2,4,index+1)
    plt.axis('off')
    plt.imshow(image ,cmap= plt.cm .gray_r,interpolation= 'nearest')#灰度图
    #通常图片都是由RGB组成，一块一块的，详见我的数字图像处理系列博客，这里想把某块显示成一种颜色，则需要调用interpolation='nearest'参数即可
    plt.title('Training : %i' %label )

n_samples = len(digits.images)
data = digits .images.reshape((n_samples,-1))
classifier = svm.SVC(gamma= 0.001)
classifier .fit(data [:n_samples//2],digits .target[:n_samples //2])
expected = digits.target[n_samples //2:]
predicted = classifier .predict(data [n_samples //2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))#分类报告
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))#混淆矩阵，用来计算分类准确率


images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()