import pandas as pd

# print('a')
data_train = pd.read_csv('X_train.csv')
# print(data_train)
bar = pd.read_csv('ytrain.csv')

data = pd.merge(data_train, bar, on='id')
data.to_csv('train.csv', index=None)
print(data)