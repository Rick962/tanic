import pydot
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split  # 数据切分
from sklearn.feature_extraction import DictVectorizer  # 特征转换器

from sklearn.model_selection import KFold
from sklearn import naive_bayes
# from sklearn import naive_bayes
# from sklearn import naive_bayes




# 1.数据获取
# titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
titanic = pd.read_csv('titanic.csv')

titanic['embarked'].fillna(method='bfill', inplace=True)
titanic['embarked'].fillna(method='ffill', inplace=True)
titanic['home.dest'].fillna(method='bfill', inplace=True)
titanic['home.dest'].fillna(method='ffill', inplace=True)




X = titanic[['pclass', 'age', 'sex', 'embarked', 'home.dest']]  # 提取要分类的特征。一般可以通过最大熵原理进行特征选择
y = titanic['survived']

# 2.数据预处理：训练集测试集分割，数据标准化
# X['age'].fillna(X['age'].mean(), inplace=True)  # age只有633个，需补充，使用平均数或者中位数都是对模型偏离造成最小的策略
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)  # 将数据进行分割

vec = DictVectorizer(sparse=False)
X = vec.fit_transform(X.to_dict(orient='record'))
# X_test = vec.transform(X_test.to_dict(orient='record'))  # 对测试数据的特征进行提取

kf = KFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    clf = DecisionTreeClassifier(max_depth=3)

    # clf = naive_bayes.BernoulliNB()
    # clf = naive_bayes.GaussianNB()
    # clf = naive_bayes.MultinomialNB()
    # clf = naive_bayes.ComplementNB()

    clf.fit(X_train, y_train)

    # y_predict = clf.predict(X_test)

    # 4.获取结果报告
    print("train: ", clf.score(X_train, y_train)*100)
    print('Accracy: ', clf.score(X_test, y_test)*100)
    # print(classification_report(y_predict, y_test, target_names=['died', 'servived']))
    # tree.export_graphviz(dtc, out_file="tree.dot", class_names=['存活', '死亡'], feature_names=vec.feature_names_,
    #                      impurity=False, filled=True)

    # tree.plot_tree(dtc, max_depth=3, feature_names=None,
    #               class_names=None, label='all', filled=False,
    #               impurity=True, node_ids=False,
    #               proportion=False, rotate=False, rounded=False,
    #               precision=3, ax=None, fontsize=None)
