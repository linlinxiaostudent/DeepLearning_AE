from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
digits = load_digits()
digit_data = digits.data
sub_data = digit_data[0:100,:]
print (sub_data.shape)
fig, axe = plt.subplots(1,12,subplot_kw=dict(xticks=[], yticks=[]))

for i in range(0,12):
    axe[i].imshow(sub_data[i,:].reshape((8,8)),cmap=plt.cm.binary, interpolation='nearest')

plt.show()

digit_pca = PCA(n_components=36,copy=True,whiten=False)
new_data = digit_pca.fit_transform(sub_data);
print(new_data.shape)

fig, axe = plt.subplots(1,12,subplot_kw=dict(xticks=[], yticks=[]))
for i in range(0,12):
    axe[i].imshow(new_data[i,:].reshape((6,6)),cmap=plt.cm.binary, interpolation='nearest')

plt.show()
