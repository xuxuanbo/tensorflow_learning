# -*- coding: utf-8 -*-
import tensorflow as tf

X = tf.placeholder(dtype=tf.float32,shape=[None,None],name="x_train")
Y = tf.placeholder(dtype=tf.float32,shape=[None,None],name="y_train")

W = tf.Variable(tf.random_normal([4,2]),name="weight")
b = tf.Variable(tf.random_normal([1,2]),name="bias")

hypothesis = tf.matmul(X,W) + b

cost = tf.reduce_mean(tf.square(hypothesis-Y))

optimizer = tf.train.AdagradOptimizer(learning_rate=.001)
train_step = optimizer.minimize(cost)


with tf.Session() as sess:
    for step in range(2001):
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.initialize_all_variables())  比较旧一点的初始化变量方法
        cost_val,W_val,b_wal =  sess.run([cost,W,b],feed_dict={X:[[2,3,4,5],[5,6,7,8]],Y:[[0.4,0.6],[0.1,0.9]]})
        print cost_val,W_val,b_wal