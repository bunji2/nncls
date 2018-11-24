import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

data_dir = "./data_set"

#mnist = input_data.read_data_sets(data_dir, one_hot=True)
mnist = input_data.read_data_sets(data_dir, one_hot=False)

#print(len(mnist.train.images[0])) # ===> 784
#print(len(mnist.train.labels)) # ===> 55000
#print(mnist.train.num_examples)   # ===> 55000

#print(type(mnist.train.images))
#print(type(mnist.train.labels))
#mnist.train.images.dtype = 'float32'
#mnist.train.labels.dtype = 'float32'

train_x = mnist.train.images
train_y = mnist.train.labels

if len(train_x) > len(train_y):
    train_x = train_x[0:len(train_y)]
elif len(train_x) < len(train_y):
    train_y = train_y[0:len(train_x)]

#mnist.test.images.dtype = 'float32'
#mnist.test.labels.dtype = 'float32'

test_x = mnist.test.images
test_y = mnist.test.labels

if len(test_x) > len(test_y):
    test_x = test_x[0:len(test_y)]
elif len(test_x) < len(test_y):
    test_y = test_y[0:len(test_x)]


import csv
with open('test.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
    for tx,ty in zip(test_x, test_y):
        l = []
        l.extend(tx)
        l.append(ty)
        writer.writerow(l)     # list（1次元配列）の場合

#for tx,ty in zip(train_x, train_y):
#    print(",".join(map(str, tx)), ",", ty)

print(train_x[0])