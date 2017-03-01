from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

# Parameters
n_hidden = 32 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)
n_blue_first = 100;

# Parameters
learning_rate = 0.0003
training_iters = 100000
batch_size = 50
display_step = 10

image_size = 784
pixels = tf.placeholder(tf.float32, [None, image_size], 'pixels')  # [batch_size, 784]
y = tf.placeholder(tf.float32, [None, 10])  # [batch_size, 10]

# Define weights
weights = {
    'blue_first': tf.Variable(tf.random_normal([n_hidden, n_blue_first])),
    'blue_second': tf.Variable(tf.random_normal([n_blue_first, n_classes]))
}
biases = {
    'blue_first': tf.Variable(tf.random_normal([n_blue_first])),
    'blue_second': tf.Variable(tf.random_normal([n_classes]))
}

def green_block(x):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    # Get lstm cell output

    import code; code.interact(local=dict(globals(), **locals()))
    outputs, _ = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return outputs[-1]

def red_block(x,weights,biases):
    import code; code.interact(local=dict(globals(), **locals()))
    return tf.matmul(x,weights['red']) + biases['red']

def blue_block(x,weights,biases):
    first_linear_layer = tf.matmul(x,weights['blue_first']) + biases['blue_first']
    relu_output = tf.nn.relu(first_linear_layer)
    second_linear_layer = tf.matmul(relu_output,weights['blue_second']) + biases['blue_second']
    return second_linear_layer

def binarize(images, threshold=0.1):
    return (threshold < images).astype('float32')

def RNN(x,weights,biases):
    rnn_inputs = tf.reshape(pixels, [-1, image_size, 1])
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    batch_encoder_outputs, _ = tf.nn.dynamic_rnn(lstm_cell, rnn_inputs, dtype=tf.float32) # [batch_size, 784, lstm_cell_size]

    # Get last time step output
    batch_outputs = batch_encoder_outputs[:, image_size-1, :]

    blue_output = blue_block(batch_outputs,weights,biases)

    return blue_output

# Define loss and optimizer
pred = RNN(pixels, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # Binarize images
        batch_x = binarize(batch_x)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={pixels: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={pixels: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={pixels: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    x_test = binarize(mnist.validation.images)
    y_test = mnist.validation.labels
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={pixels: x_test, y: y_test}))
