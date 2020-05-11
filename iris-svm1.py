from matplotlib.colors import ListedColormap
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt


# 定义画图的方法：

def plot_decision_regions(x, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')  # 形状
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')  # 颜色
    cmap = ListedColormap(colors[:len(np.unique(y))])  #

    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1  # 特征1的最大值，最小值，作为坐标的区域
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))  # 他用画无数个点给画布添颜色

    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)  # 把均匀分布的x，y拉平，然后在竖起来 用模型预测 y 值
    z = z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)  # alpha:透明度，cmap：分类的颜色
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        # 给每一类画点，alpha：点的透明度，c:颜色，marker:形状， label:标签
        plt.scatter(x=x[y == cl, 0], y=x[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    if test_idx:
        x_test, y_test = x[test_idx, :], y[test_idx]
        plt.scatter(x_test[:, 0], x_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')


# 加载数据

iris = datasets.load_iris()
x = iris.data[:, [1, 2]]  # iris里有4个特征，只保留petal width（花瓣宽度）, petal length（花瓣长度）特征分析
y = iris.target  # 有3个种类
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)  # fit 并 transform
# sc.mean_
# sc.scale_  # x 只有两个维度，调整后的均值，和标准差只有两个
x_test_sc = sc.transform(x_test)
x_combined = np.vstack((x_train_sc, x_test_sc))  # 纵向合并
y_combined = np.hstack((y_train, y_test))  # 横向合并


# 加载 SVC kernel=linear

svm1 = SVC(kernel='linear', C=0.1, random_state=1)
svm2 = SVC(kernel='linear', C=10, random_state=1)  # 线性核函数中，c 是 对错误值的敏感程度
svm1.fit(x_train_sc, y_train)
svm2.fit(x_train_sc, y_train)

fig = plt.figure(figsize=(10, 12))
ax1 = fig.add_subplot(2, 2, 1)
plot_decision_regions(x_combined, y_combined, classifier=svm1)
# plt.xlabel('petal length [standardized')
plt.ylabel('petal width [standardized')
plt.title('c = 0.1')

ax2 = fig.add_subplot(2, 2, 2)
plot_decision_regions(x_combined, y_combined, classifier=svm2)
# plt.xlabel('petal length [standardized')
plt.ylabel('petal width [standardized')
plt.title('c = 10')

# 加载 SVC kernel=rbk 径向基核函数 高斯核函数

svm1 = SVC(kernel='rbf', C=1, gamma=0.1, random_state=1)
svm2 = SVC(kernel='rbf', C=1, gamma=10, random_state=1)  # 线性核函数中，c 是 对错误值的敏感程度
svm1.fit(x_train_sc, y_train)
svm2.fit(x_train_sc, y_train)


ax3 = fig.add_subplot(2, 2, 3)
plot_decision_regions(x_combined, y_combined, classifier=svm1)
plt.xlabel('petal length [standardized')
plt.ylabel('petal width [standardized')
plt.title('gamma = 0.1')

ax4 = fig.add_subplot(2, 2, 4)
plot_decision_regions(x_combined, y_combined, classifier=svm2)
plt.xlabel('petal length [standardized')
plt.ylabel('petal width [standardized')
plt.title('gamma = 10')


plt.show()
