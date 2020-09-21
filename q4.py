import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def xlsx_to_csv_pd():
    '''xlsx 转 csv'''
    data_xlsx = pd.read_excel('./t2.xlsx', sheet_name=None,
    index_col=0)
    for key in data_xlsx:
        data_xlsx[key].to_csv('./t2_' + key + '.csv', encoding='utf-8')

data2 = pd.read_csv('./t2_2.csv')
data3 = pd.read_csv('./t2_3.csv')
data4 = pd.read_csv('./t2_4.csv')
data5 = pd.read_csv('./t2_5.csv')
data6 = pd.read_csv('./t2_6.csv')

raw_data = data2.append([data3, data4, data5, data6])


data = raw_data.iloc[:,1:5]
label = raw_data.iloc[:,0]

train_data,test_data,train_label,test_label=train_test_split(data,label,random_state=100,test_size=0.3)

# svm 支持向量机
# result = np.zeros(shape=(5, 4))
# C = [0.01, 0.1, 1, 10, 100]
# for i in range(len(C)):
#     a = []
#     for g in [0.01, 0.1, 1, 10]:
#         classifier = svm.SVC(C=C[i], kernel='rbf', gamma=g, decision_function_shape='ovr')
#         classifier.fit(train_data, train_label)
#         pre_label = classifier.predict(test_data) # test_data -> train_data
#         a.append(accuracy_score(test_label, pre_label))
#     result[i] = a
# print(result) # 这是测试集的结果

# num = [0.01, 0.1, 1, 10]
# num2 = [0.01, 0.1, 1, 10, 100]
# fig = plt.figure()
# ax = plt.subplot(111)
# for i in range(5):
#     ax.plot(num, result[i], label=num2[i])
# plt.legend()
# plt.xlabel('gamma')
# plt.ylabel('Accuracy')
# plt.title('gamma - Accuracy')
# plt.show()

# # 决策树
from sklearn.tree import DecisionTreeClassifier
result = []
deep = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
for i in deep:
    clf = DecisionTreeClassifier(max_depth=i, random_state=0)
    clf.fit(train_data, train_label)
    pre_label = clf.predict(test_data)
    result.append(accuracy_score(test_label, pre_label))
print(result)
plt.plot(deep, result)
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Max Depth - Accuracy')
plt.show()


# bp
# from sklearn.neural_network import MLPClassifier
# mlp = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(10, 10, 10), random_state=1)

# mlp.fit(train_data, train_label)
# pre_label = mlp.predict(test_data)
# print(accuracy_score(test_label, pre_label))


