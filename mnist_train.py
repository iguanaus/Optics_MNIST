#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from time import time
import numpy as np

# Download the dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#############################
#######Hyperparameters#######
#############################

n_per_layer =       1000   #Number neurons per layer
#Structure is input -> first layer -> second layer -> output
#                728       n_per            n_per        10
#    Non linearity of relu between the layers.
n_iter =            5    #Number training iterations
save_filename = '/Users/johnpeurifoy/Documents/skewl/PhotoNet_MNIST/mnist_fc_best_'+str(n_per_layer) #Where to save the file
output_folder = '/Users/johnpeurifoy/Documents/skewl/PhotoNet_MNIST/'+str(n_per_layer)+'_layer/'

def save_weights(weights,biases,output_folder):
    for i in xrange(0, len(weights)):
        weight_i = weights[i].eval()
        np.savetxt(output_folder+"w_"+str(i)+".txt",weight_i,delimiter=',')
        bias_i = biases[i].eval()
        np.savetxt(output_folder+"b_"+str(i)+".txt",bias_i,delimiter=',')
        print("Bias: " , i, " : ", bias_i)
    return


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


# correct labels
y_ = tf.placeholder(tf.float32, [None, 10])

# input data
x = tf.placeholder(tf.float32, [None, 784])

# build the network

W_fc1 = weight_variable([784, n_per_layer])
b_fc1 = bias_variable([n_per_layer])
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

W_fc2 = weight_variable([n_per_layer, n_per_layer])
b_fc2 = bias_variable([n_per_layer])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

W_fc3 = weight_variable([n_per_layer, 10])
b_fc3 = bias_variable([10])
y = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)


# define the loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# define training step and accuracy
train_step = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# create a saver
saver = tf.train.Saver()

with tf.Session() as sess:

    # initialize the graph
    init = tf.initialize_all_variables()
    #sess = tf.Session()
    sess.run(init)

    # train
    batch_size = 100
    print("Startin Burn-In...")
    saver.save(sess, save_filename)
    for i in range(700):
        input_images, correct_predictions = mnist.train.next_batch(batch_size)
        if i % (60000/batch_size) == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                x: input_images, y_: correct_predictions})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            # validate
            test_accuracy = sess.run(accuracy, feed_dict={
                x: mnist.test.images, y_: mnist.test.labels})
            print("Validation accuracy: %g." % test_accuracy)
        sess.run(train_step, feed_dict={x: input_images, y_: correct_predictions})

    saver.restore(sess, save_filename)
    print("Starting the training...")
    start_time = time()
    best_accuracy = 0.0
    for i in range(int(n_iter*60000/batch_size)):
        input_images, correct_predictions = mnist.train.next_batch(batch_size)
        if i % (60000/batch_size) == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                x: input_images, y_: correct_predictions})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            # validate
            test_accuracy = sess.run(accuracy, feed_dict={
                x: mnist.test.images, y_: mnist.test.labels})
            if test_accuracy >= best_accuracy:
                saver.save(sess, save_filename)
                best_accuracy = test_accuracy
                print("Validation accuracy improved: %g. Saving the network." % test_accuracy)
            else:
                saver.restore(sess, save_filename)
                print("Validation accuracy was: %g. It was better before: %g. " % (test_accuracy, best_accuracy) +
                      "Using the old params for further optimizations.")
        sess.run(train_step, feed_dict={x: input_images, y_: correct_predictions})

    weights = [W_fc1,W_fc2,W_fc3]

    biases = [b_fc1,b_fc2,b_fc3]

    save_weights(weights,biases,output_folder)

print("The training took %.4f seconds." % (time() - start_time))

# validate
print("Best test accuracy: %g" % best_accuracy)