# import pydot
# import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer  # 特征转换器
from sklearn import model_selection
from sklearn import svm
from sklearn import naive_bayes
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import BernoulliRBM, MLPClassifier, MLPRegressor

# 1.数据获取
# titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# titanic.to_csv('titanic.csv')
titanic = pd.read_csv('titanic.csv')

# 填充缺失数据
titanic['embarked'].fillna(method='bfill', inplace=True)
titanic['embarked'].fillna(method='ffill', inplace=True)
titanic['home.dest'].fillna(method='bfill', inplace=True)
titanic['home.dest'].fillna(method='ffill', inplace=True)

X = titanic[['pclass', 'age', 'sex', 'embarked', 'home.dest']]
y = titanic['survived'].values

print(X)
# 特征提取
vec = DictVectorizer(sparse=False)
X = vec.fit_transform(X.to_dict(orient='record'))

kf = model_selection.KFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    # print(len(X_train), len(X_test))

    # 3.model
    # 使用决策树对测试数据进行类别预测  预剪枝法　最大深度为3
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)

    # 贝叶斯分类
    # clf = naive_bayes.BernoulliNB()
    # clf = naive_bayes.GaussianNB()
    # clf = naive_bayes.MultinomialNB()
    # clf = naive_bayes.ComplementNB()
    # clf.fit(X_train, y_train)

    # svm  kernel: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    # clf = svm.SVC(C=2,  gamma='auto', kernel='poly', degree=3)
    # clf.fit(X_train, y_train, sample_weight=None, )

    # 最小二乘法
    # clf = LinearRegression()
    # clf.fit(X_train, y_train)
    # print(X_test[0])
    # print(len(X_test[1]))

    # KNN  K:11,13时 89%
    # clf = KNeighborsClassifier(n_neighbors=13)
    # clf = RadiusNeighborsClassifier()
    # clf.fit(X_train, y_train)

    # NN
    # clf = MLPClassifier(hidden_layer_sizes=18, activation='logistic', max_iter=1500,)
    # clf.fit(X_train, y_train)

    # 4.获取结果报告
    # print('train Accracy:   ', '{:.2f}%'.format(clf.score(X_train, y_train) * 100))
    print('Accracy: ', '{:.2f}%'.format(clf.score(X_test, y_test) * 100))
    # print(classification_report(y_predict, y_test, target_names=['died', 'servived']))

    # test_one =
    #      [19,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #        0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,
    #        1,  0,]

    # asd = np.array(test_one).reshape(1, -1)
    # y_predict = clf.predict(asd)
    # print(y_predict)
