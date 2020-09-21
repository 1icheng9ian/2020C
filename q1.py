# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 13:47:26 2020

@author: 1icheng9ian
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from scipy import signal
from sklearn.decomposition import FastICA

# import matplotlib.pyplot as plt

dic = {101: (1, 7), 102: (1, 8), 103: (1, 9), 104: (1, 10), 105: (1, 11), 106: (1, 12),
       107: (2, 7), 108: (2, 8), 109: (2, 9), 110: (2, 10), 111: (2, 11), 112: (2, 12),
       113: (3, 7), 114: (3, 8), 115: (3, 9), 116: (3, 10), 117: (3, 11), 118: (3, 12),
       119: (4, 7), 120: (4, 8), 121: (4, 9), 122: (4, 10), 123: (4, 11), 124: (4, 12),
       125: (5, 7), 126: (5, 8), 127: (5, 9), 128: (5, 10), 129: (5, 11), 130: (5, 12),
       131: (6, 7), 132: (6, 8), 133: (6, 9), 134: (6, 10), 135: (6, 11), 136: (6, 12)}

def get_data(n, x, y, num, file):
    '''获取数据
    n 字符位置; x 截取起始; y 截取结束;num 1-9; file 文件; expend 是否需要扩充'''
    data = pd.read_csv(file[n], header=None)
    data = chebfilter(data)
    event = pd.read_csv(file[n+12], header=None)
    piece = cut(data, event, num, x, y)
    return piece

def cut(data, event, num, x, y):
    '''x 截取起始; y 截取结束;num 是1-12的数字
    只截一段'''
    index = event[event[0]==num]
    index = index[1].values.tolist()
    part = data[index[0]+x:index[0]+y].reset_index(drop=True)
    part = sign(part, event, num, dic)
    return part
    
def sign(data, event, num, dic):
    '''标记'''
    char_num = event.iloc[0,0]
    (a,b) = dic[char_num]
    if num == a or num == b:
        data.loc[:,20] = 1
    else:
        data.loc[:,20] = 0
    return data

def normalization(x):
    '''归一化'''
    return (x-np.mean(x, axis=0))/np.std(x, axis=0)
    
def split(raw_data, l):
    '''l 列号'''
    data = raw_data[:,0:l]
    label = raw_data[:,l]
    train_data,test_data,train_label,test_label=train_test_split(data,label,random_state=100,test_size=0.25)
    return train_data,test_data,train_label,test_label
   
def chebfilter(rawdata):
   '''预处理, IIR滤波'''
   b, a = signal.cheby1(8, 0.1, 0.03)
   for i in rawdata:
       rawdata[i] = signal.filtfilt(b, a, rawdata[i])
   return rawdata
 
def file_name(file_dir):
    '''找所有文件的名字'''
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.csv':
                L.append(os.path.join(root, file))
    return L


if __name__ == "__main__":
    p1 = file_name('./S1')
    p2 = file_name('./S2')
    p3 = file_name('./S3')
    p4 = file_name('./S4')
    p5 = file_name('./S5')
    # print(p1[43])
    
    ica = FastICA(n_components=20)
    
    train = np.zeros((1,21))
    
    file = p1
    for n in range(20, 30):
        for num in range(1,13):
            part = get_data(n, 60, 90, num, file)
            part, label = normalization(part.loc[:,0:19]), part.loc[:,20]
            label = label.values
            # 对part做ICA
            u = ica.fit_transform(part)
            part = np.column_stack((u, label))
            # 合并
            train = np.row_stack((train, part))
    train = np.delete(train,0,0)
    
    train_data,test_data,train_label,test_label = split(train, 20)
            
    # svm
    # classifier = svm.SVC(C=1, kernel='rbf', gamma=10, 
    #                       decision_function_shape='ovr')
    # classifier.fit(train_data, train_label)
    
    # pre_label = classifier.predict(test_data)
    # print(accuracy_score(test_label, pre_label))
    
    # tree
    # from sklearn.tree import DecisionTreeClassifier
    # clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    # clf.fit(train_data, train_label)
    
    # test data
    test = np.zeros((1,21))
    for n in range(0,9):
        for num in range(1,13):
            data = pd.read_csv(file[n], header=None)
            data = chebfilter(data)
            event = pd.read_csv(file[n+10], header=None)
            
            index = event[event[0]==num]
            index = index[1].values.tolist()
            part_test = data[index[0]+60:index[0]+90].reset_index(drop=True)
            part_test.loc[:,20] = 0
            
            part_test, test_label_2 = normalization(part_test.loc[:,0:19]), part_test.loc[:,20]
            test_label_2 = test_label_2.values
            u_test = ica.fit_transform(part_test)
            part_test = np.column_stack((u_test, test_label_2))
            
            test = np.row_stack((test, part_test))
    test = np.delete(test,0,0)
    test = test[:,:20]
    
    # pre_test_label = classifier.predict(test)
    
    # pre_test_label = clf.predict(test)
    
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(2, random_state=0).fit(train_data)
    pre_test_label = kmeans.predict(test_data)
    # print(accuracy_score(test_label, pre_test_label))
    
    pre_test_label_2 = kmeans.predict(test)
    result = np.column_stack((test, pre_test_label_2))
    result = pd.DataFrame(result)
    result.to_csv('./predict_1.csv', header=None, index=None)
    
            
            
            
            
            