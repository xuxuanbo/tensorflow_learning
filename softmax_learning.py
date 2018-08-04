# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('/home/hadoopnew/下载/mnist/',one_hot=True)

x = tf.placeholder(dtype=tf.float32,shape=[None,784],name='x_train')
y = tf.placeholder(dtype=tf.float32,shape=[None,10],name='y_train')

w = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.zeros([10]))

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

cross_entropy = -tf.reduce_sum(y*tf.log(hypothesis))
# cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=hypothesis))

global_step = tf.Variable(0)

learning_rate = tf.train.exponential_decay(0.01,global_step=global_step,decay_steps=100,decay_rate=0.96,staircase=True)

train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(15000):
        batch_x ,batch_y = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x:batch_x,y:batch_y})
        if step % 500 == 0:
            accuracy =tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hypothesis,1),tf.argmax(y,1)),"float"))
            print '第',step,'步:',sess.run(accuracy,feed_dict={x: mnist.test.images, y: mnist.test.labels})

