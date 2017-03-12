import tensorflow as tf
import numpy as np

from binaryop import binarize_weights, binary_activation
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, padding='SAME'):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def batch_norm_layer(x, name):
    with tf.name_scope(name) as scope:
        mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        norm = tf.nn.batch_normalization(x, mean, variance, None, None, 1e-5)
        return norm

# Build Computational Graph
sess = tf.InteractiveSession()

# Initialize placeholders for data & labels
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

# reshape to make image volumes
x_image = tf.reshape(x, [-1,28,28,1])

# First Conv Layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
BW_conv1 = binarize_weights(W_conv1)

norm1 = batch_norm_layer(x_image)
binAct1 = binary_activation(norm1)

h_conv1 = tf.nn.relu(conv2d(binAct1, BW_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
BW_conv2 = binarize_weights(W_conv2)

norm2 = batch_norm_layer(h_pool1)
binAct2 = binary_activation(norm2)

h_conv2 = conv2d(binAct2, BW_conv2) + b_conv2
h_relu2 = tf.nn.relu(h_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# fully connected layer
W_fc1 = weight_variable([7, 7, 64, 1024])
b_fc1 = bias_variable([1024])
BW_fc1 = binarize_weights(W_fc1)

norm3 = batch_norm_layer(h_pool2)
binAct3 = binary_activation(norm3)

h_fc1 = tf.nn.relu(conv2d(binAct3, BW_fc1, padding='VALID') + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
W_fc2 = weight_variable([1, 1, 1024, 10])
b_fc2 = bias_variable([10])
BW_fc2 = binarize_weights(W_fc2)

norm4 = batch_norm_layer(h_fc1_drop)
binAct4 = binary_activation(norm3)

y_conv = tf.reshape(conv2d(binAct4, BW_fc2) + b_fc2, [-1, 10])

# create train ops
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize all variables
sess.run(tf.global_variables_initializer())

# train loop
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i % 100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
