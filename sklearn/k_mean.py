from sklearn import cluster,datasets
iris = datasets .load_iris ()
X_iris = iris.data
y_iris = iris.target

k_means = cluster.KMeans (n_clusters= 3)
k_means .fit(X_iris )
y_iris_k_means = k_means .labels_ [::10]
print('K-means聚类结果：') 
print(y_iris_k_means )
print('数据集原始标签：')
print(y_iris [::10])