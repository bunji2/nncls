import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
mnist = datasets.fetch_mldata('MNIST original')
X,y = mnist.data,mnist.target
X = X/255.

indices = np.random.randint(len(X), size=int(len(X)/2))
train_x = X[indices]
train_y = y[indices]
indices = np.random.randint(len(X), size=int(len(X)/2))
test_x = X[indices]
test_y = y[indices]

#print(X[0], y[0])
#print(test_x[0], test_y[0])
import csv
with open('test.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
    for tx,ty in zip(test_x, test_y):
        l = []
        l.extend(tx)
        l.append(ty)
        writer.writerow(l)     # list（1次元配列）の場合