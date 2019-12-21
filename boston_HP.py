from sklearn.datasets import load_boston
import pandas as pd

from sklearn.feature_extraction import DictVectorizer  # 特征转换器

from sklearn import model_selection

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn import neural_network
# data, target = load_boston(return_X_y=True)

df = pd.read_csv('boston.csv')

data = df[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']]
label = df['label']

vec = DictVectorizer(sparse=False)
X = vec.fit_transform(data.to_dict(orient='record'))
# print(data)
kf = model_selection.KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, y_train = X[train_index], label[train_index]
    X_test, y_test = X[test_index], label[test_index]

    # print(X_train[9])

    # clf = LinearRegression()
    # clf.fit(X_train, y_train)
    # print(clf.score(X_test, y_test))

    # 90
    # clf = DecisionTreeRegressor()
    # clf.fit(X_train, y_train)
    # print(clf.score(X_test, y_test))

    # n_estimators 迭代次数　一般为10  大易过拟合　小易欠
    clf = RandomForestRegressor(max_depth=10, n_estimators=5)
    clf.fit(X_train, y_train)


    # activation: 'identity', 'logistic', 'tanh', 'relu'
    # clf = neural_network.MLPRegressor(hidden_layer_sizes=50, activation='relu', max_iter=2000,)
    # clf.fit(X_train, y_train)
    # print(clf.score(X_test, y_test))

    # print(X_test[0])
    # print(len(X_test[1]))

    print(clf.score(X_test, y_test))