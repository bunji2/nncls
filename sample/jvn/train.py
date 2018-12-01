# https://nilsmagnus.github.io/post/go-tensorflow/

import numpy as np
import tensorflow as tf
from data import (train_x, train_y, test_x, test_y)

# next_batch : data/label の中から batch_size の個数だけランダムで抜き出す関数
def next_batch(data, label, batch_size):
    num = len(data)
    if len(data) > len(label):
        num = len(label)
    indices = np.random.randint(num, size=batch_size)
    #print("inices =", indices)
    #print("len(data) =", len(data))
    #print("len(label) =", len(label))
    return data[indices], label[indices]

# nn_train : NN を用いた学習
def nn_train(num_input, num_output, batch_size=20, epoch=200, out_prefix='./model'):
    # 入力層
    x = tf.placeholder(tf.float32, [None, num_input], name='INPUT')

    #線形回帰
    W = tf.Variable(tf.zeros([num_input, num_output]))
    b = tf.Variable(tf.zeros([num_output]))
    y = tf.add(tf.matmul(x, W) , b, name="OUTPUT")

    ## 中間層1
    #hidden1 = tf.layers.dense(inputs=x, units=100, activation=tf.nn.relu, name="hidden1")
    ## 中間層2
    #hidden2 = tf.layers.dense(inputs=hidden1, units=50, activation=tf.nn.relu, name="hidden2")
    #y = tf.layers.dense(inputs=hidden2, units=num_output, activation=tf.nn.sigmoid, name="OUTPUT")

    # 教師データ
    labels = tf.placeholder(tf.float32, [None, num_output], name='teacher_signal')
    y_ = labels
    #y_ = tf.one_hot(labels, depth=num_output, dtype=tf.float32)

    # 教師値と NN の予測値の比較
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))

    # 学習結果の評価
    #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #rrr = tf.reduce_sum(tf.squared_difference(y, y_))
    # 学習の実行

    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    #batch_size = 20
    #epoch = 200

    #batches_in_epoch = training_data.shape[0]
    batches_in_epoch = train_x.shape[0]

    with tf.Session() as sess:

        with tf.name_scope('summary'):
            tf.summary.scalar('loss', loss)
            #tf.summary.scalar('accuracy', accuracy)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('./logs', sess.graph)


        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            for j in range(batches_in_epoch):
                batch_train_x, batch_train_y = next_batch(train_x, train_y, batch_size)
                sess.run(train_op, feed_dict={x: batch_train_x, labels: batch_train_y})
            print(i, sess.run(loss, feed_dict={x: test_x, labels: test_y}))
            #print(i, sess.run(loss, feed_dict={x: test_x, labels: test_y}), sess.run(rrr, feed_dict={x: test_x, labels: test_y}))
        # モデルデータの保存
        builder = tf.saved_model.builder.SavedModelBuilder(out_prefix)
        builder.add_meta_graph_and_variables(sess,["serve"])
        builder.save()

        writer.close()

def run():
    #print(train_x[0], train_y[0])
    #return
    num_input = 741
    num_output = 13
    batch_size = 40
    epoch = 200
    nn_train(num_input, num_output, batch_size, epoch)

if __name__ == '__main__':
    run()