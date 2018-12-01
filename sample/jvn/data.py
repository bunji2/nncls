# -*- coding: utf-8 -*-

import numpy as np
import json

def foo(indices, num_words):
    r = np.zeros(num_words)
    for idx in indices:
        r[idx] = r[idx] + 1
    return r

def split_data(x, y, num):
    indices = np.random.randint(x.shape[0], size=num)
    indices2 = np.random.randint(x.shape[0], size=(len(x)-num))
    return x[indices], y[indices], x[indices2], y[indices2]

data_dir = "./data_set"

f = open(data_dir+'/words.json', 'r',encoding="utf-8")
docs = json.load(f)

print(docs["note"])
num_words = int(docs["wordcount"])
print("num_words =", num_words)

x = []
y = []
for doc in docs["docs"]:
    x.append(foo(doc["seq"], num_words))
    y.append(foo(doc["labels"], 13))

x = np.array(x, dtype='float32')
y = np.array(y, dtype='float32')
#print(x.shape)
#print(y.shape)

train_x, train_y, test_x, test_y = split_data(x, y, int(3*len(x)/4))
#print(train_x.shape)
#print(train_y.shape)

print("saving test.csv...")
import csv
with open('test.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
    for tx,ty in zip(test_x, test_y):
        l = []
        l.extend(tx)
        l.extend(ty)
        writer.writerow(l)
