from datetime import datetime
now = datetime.now()
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# import dataset with one-hot class encoding
mnist = input_data.read_data_sets("../MINST_data", one_hot=True)

def binarize(images, threshold=0.1):
    return (threshold < images).astype('float32')

image_size = 784
pixels = tf.placeholder(tf.float32, [None, image_size], 'pixels')  # [batch_size, 784]
y = tf.placeholder(tf.float32, [None, 10])  # [batch_size, 10]
learning_rate = tf.placeholder(tf.float32)
lstm_cell_size = 128

def rnn_lstm_graph(lstm_cell_size):
    # Turn pixels into a 3D tensor
    rnn_inputs = tf.reshape(pixels, [-1, image_size, 1]) # [batch_size, 784, 1]
    # Build RNN graph
    with tf.variable_scope('enc'):
        lstm_cell = tf.contrib.rnn.LSTMCell(lstm_cell_size, state_is_tuple=True)

    batch_encoder_outputs, _ = tf.nn.dynamic_rnn(lstm_cell, rnn_inputs, dtype=tf.float32) # [batch_size, 784, lstm_cell_size]

    # Get last time step output
    batch_outputs = batch_encoder_outputs[:, image_size-1, :]  # [batch_size, lstm_cell_size]

    # Put through linear layers with relu
    # [batch_size, lstm_cell_size] -> [batch_size, 100]
    with tf.variable_scope('out1'):
        W1 = tf.Variable(tf.random_normal([lstm_cell_size, 100]))
        b1 = tf.Variable(tf.random_normal([100]))
        h1 = tf.nn.relu(tf.matmul(batch_outputs, W1) + b1)

    # Put through linear layers with relu
    # [batch_size, 100] -> [batch_size, 10]
    with tf.variable_scope('out2'):
        W2 = tf.Variable(tf.random_normal([100, 10]))
        b2 = tf.Variable(tf.random_normal([10]))
        logits = tf.matmul(h1, W2) + b2

    return logits

logits = rnn_lstm_graph(lstm_cell_size)
# Apply softmax and cross entropy to get loss
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, y))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Manually apply softmax and then take the max to get prediction
pred = tf.arg_max(tf.nn.softmax(logits), 1)

# Calclulate accuracy
correct_prediction = tf.equal(pred, tf.argmax(y, 1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('task1/lstm128/%s/' % now.strftime("%Y%m%d-%H%M%S"), graph=tf.get_default_graph())


def train_model(lr):
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        # Train
        best_accuracy = 0
        for epoch in range(300):
            for j in range(220):
                batch_xs, batch_ys = mnist.train.next_batch(250)
                batch_xs = binarize(batch_xs)
                train_feed_dict = {
                                pixels: batch_xs,
                                y: batch_ys,
                                learning_rate: lr
                             }
                _ = sess.run([train_step], feed_dict=train_feed_dict)

            x_test = binarize(mnist.validation.images)
            y_test = mnist.validation.labels
            test_feed_dict = {
                pixels: x_test,
                y: y_test,
                learning_rate: lr
            }
            acc, summary = sess.run([accuracy, merged], feed_dict=test_feed_dict)
            print('Test Accuracy for learning rate %s at epoch %s: %s' % (
            lr, epoch, acc))
            train_writer.add_summary(summary, epoch)

            if epoch%10 == 0:
                if acc>best_accuracy:
                    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
                    saver.save(sess, 'task1/lstm128/test_model')
                    best_accuracy = acc

if __name__ == "__main__":
    learning_rates = 0.0003
    train_model(learning_rates)
