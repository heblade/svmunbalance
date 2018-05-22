import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sklearn.svm as svm

def startjob():
    warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
    np.random.seed(0) #保持每次生成的数据相同

    c1 = 990
    c2 = 10
    N = c1 + c2
    x_c1 = 3 * np.random.randn(c1, 2)
    x_c2 = 0.5 * np.random.randn(c2, 2) + (4, 4)
    x = np.vstack((x_c1, x_c2))
    y = np.ones(N)
    y[:c1] = -1

    #显示大小
    s = np.ones(N) * 30
    s[:c1] = 10

    #分类器
    clfs = [svm.SVC(C=1, kernel='linear'),
            svm.SVC(C=1, kernel='linear', class_weight={-1:1, 1:50}),
            #根据结果来看，数量很小的样本权重取99与10，结果其实变化不大
            svm.SVC(C=0.8, kernel='rbf', gamma=0.5, class_weight={-1:1, 1:99}),
            svm.SVC(C=0.8, kernel='rbf', gamma=0.5, class_weight={-1:1, 1:10})]
    titles = 'Linear', 'Linear, Weight=50', 'RBF, Weight=99', 'RBF, Weight=10'
    x1_min, x2_min = np.min(x, axis=0)
    x1_max, x2_max = np.max(x, axis=0)
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    grid_test = np.stack((x1.flat, x2.flat), axis=1) #测试点
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 8), facecolor='w')
    for i, clf in enumerate(clfs):
        clf.fit(x, y)
        y_hat = clf.predict(x)
        print(i+1, '次: ')
        print('accuracy: \t', accuracy_score(y, y_hat))
        print('precision: \t', precision_score(y, y_hat, pos_label=1))
        print('recall: \t', recall_score(y, y_hat, pos_label=1))
        print('F1-score: \t', f1_score(y, y_hat, pos_label=1))
        plt.subplot(2, 2, i+1)
        grid_hat = clf.predict(grid_test)
        grid_hat = grid_hat.reshape(x1.shape)
        plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light, alpha=0.8)
        plt.scatter(x[:,0], x[:,1], c=y, edgecolors='k', s=s, cmap=cm_dark)
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.title(titles[i])
        plt.grid()
    plt.suptitle('不平衡数据的处理', fontsize=18)
    plt.tight_layout(1.5)
    plt.subplots_adjust(top=0.92)
    plt.show()



if __name__ == '__main__':
    startjob()