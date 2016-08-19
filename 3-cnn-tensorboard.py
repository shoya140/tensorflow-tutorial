# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

def bias_variable(shape, name):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)

def conv2d(x, W, name):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def inference(x, keep_prob):
    with tf.name_scope("inference") as scope:
        x_image = tf.reshape(x, [-1, 28, 28, 1])

        with tf.name_scope("conv1") as scope:
            W_conv1 = weight_variable([5, 5, 1, 32], name="W_conv1")
            b_conv1 = bias_variable([32], name="b_conv1")
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, name="conv1") + b_conv1)

        with tf.name_scope("pool1") as scope:
            h_pool1 = max_pool_2x2(h_conv1, name="pool1")

        with tf.name_scope("conv2") as scope:
            W_conv2 = weight_variable([5, 5, 32, 64], name="W_conv2")
            b_conv2 = bias_variable([64], name="b_conv2")
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, name="conv2") + b_conv2)

        with tf.name_scope("pool2") as scope:
            h_pool2 = max_pool_2x2(h_conv2, name="pool2")

        with tf.name_scope("full1") as scope:
            W_fc1 = weight_variable([7*7*64, 1024], name="W_fc1")
            b_fc1 = bias_variable([1024], name="b_fc1")
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        with tf.name_scope("full2"):
            W_fc2 = weight_variable([1024, 10], name="W_fc2")
            b_fc2 = bias_variable([10], name="b_fc2")
            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return y_conv

def loss(output, y_):
    with tf.name_scope("loss") as scope:
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output), reduction_indices=[1]))
        tf.scalar_summary("entropy", cross_entropy)

        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary("accuracy", accuracy)
    return cross_entropy, accuracy

def training(loss):
    with tf.name_scope("training") as scope:
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    return train_step

def feed_dict(train):
    if train:
        xs, ys = mnist.train.next_batch(50)
        k = 0.5
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, shape=[None, 784], name="input")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="output")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    output = inference(x, keep_prob)
    entropy, accuracy = loss(output, y_)
    training_op = training(entropy)

    with tf.Session() as sess:
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter('log/train', graph=sess.graph)
        test_writer = tf.train.SummaryWriter('log/test')

        sess.run(tf.initialize_all_variables())

        for i in range(500):
            if i % 10 == 0:
                summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
                test_writer.add_summary(summary, i)
            else:
                summary, acc = sess.run([merged, training_op], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)