# coding: utf-8

import tensorflow as tf
import numpy as np
import random
from datetime import datetime as dt

INPUT_N = 1
HIDDEN_N = 100
OUTPUT_N = 1
LENGTH_OF_SEQUENCES = 10
SIZE_OF_MINI_BATCH = 100
FORGET_BIAS = 0.8

SAMPLES_N = 1000

# creating sample data
X = np.zeros((SAMPLES_N, LENGTH_OF_SEQUENCES))
for row_idx in range(SAMPLES_N):
    X[row_idx,:] = np.around(np.random.rand(LENGTH_OF_SEQUENCES)).astype(int)
t = np.sum(X, axis=1)

def get_batch(batch_size, X, t):
    rnum = [random.randint(0, len(X)-1) for x in range(batch_size)]
    xs = np.array([[[y] for y in list(X[r])] for r in rnum])
    ts = np.array([[t[r]] for r in rnum])
    return xs,ts

def weight_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

def bias_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

def inference(x, instate):
    with tf.name_scope("inference") as scope:

        W1 = weight_variable([INPUT_N, HIDDEN_N], name="weight1")
        W2 = weight_variable([HIDDEN_N, OUTPUT_N], name="weight1")
        b1 = bias_variable([HIDDEN_N], name="bias1")
        b2 = bias_variable([OUTPUT_N], name="bias2")

        in1 = tf.transpose(x, [1, 0, 2])
        in2 = tf.reshape(in1, [-1, INPUT_N])
        in3 = tf.matmul(in2, W1) + b1
        in4 = tf.split(0, LENGTH_OF_SEQUENCES, in3)

        cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_N, forget_bias=FORGET_BIAS)
        rnn_output, states_op = tf.nn.rnn(cell, in4, initial_state=instate)
        output = tf.matmul(rnn_output[-1], W2) + b2
    return output

def loss(output_op, y_):
    with tf.name_scope("loss") as scope:
        loss_op = tf.reduce_mean(tf.square(output_op - y_))
        tf.scalar_summary("loss", loss_op)
    return loss_op

def training(loss_op):
    with tf.name_scope("training") as scope:
        train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(loss_op)
    return train_op

def feed_dict():
    xs, ys = get_batch(SIZE_OF_MINI_BATCH, X, t)
    return {x: xs, y_: ys, instate: np.zeros((SIZE_OF_MINI_BATCH, HIDDEN_N*2))}

with tf.Graph().as_default():
    x       = tf.placeholder(tf.float32, [None, LENGTH_OF_SEQUENCES, INPUT_N], name="input")
    y_      = tf.placeholder(tf.float32, [None, OUTPUT_N], name="label")
    instate = tf.placeholder(tf.float32, [None, HIDDEN_N*2], name="instate")

    output_op = inference(x, instate)
    loss_op = loss(output_op, y_)
    training_op = training(loss_op)

    with tf.Session() as sess:
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter("log/train/"+dt.now().strftime("%Y-%m-%d_%H-%M-%S"), graph=sess.graph)

        sess.run(tf.initialize_all_variables())

        for i in range(1000):
            sess.run(training_op, feed_dict=feed_dict())
            if i % 100 == 0:
                summary_str, train_loss = sess.run([summary_op, loss_op], feed_dict=feed_dict())
                print("train#%d, train loss: %e" % (i, train_loss))
                summary_writer.add_summary(summary_str, i)