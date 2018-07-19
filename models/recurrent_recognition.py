import tensorflow as tf
from hypergradient.Adam_HD_optimizer import AdamHDOptimizer
from hypergradient.SGDN_HD_optimizer import MomentumSGDHDOptimizer

from tensorflow.contrib import rnn

class RecurrentRecognition:
    def __init__(self, learning_rate=1e-4):
        self.inputs = tf.placeholder(tf.float32, [None, 513, 469, 1])
        self.labels = tf.placeholder(tf.int32, [None])
        self.dropout_rate = tf.placeholder(tf.float32)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        splitted = tf.split(self.inputs[:, :, 0:468, :], 18, axis=2)
        lstm = rnn.BasicLSTMCell(128)


        self.readout = tf.layers.dense(None, 2)

        # print the shapes of each layer
        self.print_convolution(self.readout)

        self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.readout)
        self.optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=self.global_step)
        #self.optimize = MomentumSGDHDOptimizer(alpha_0=learning_rate).minimize(self.loss, self.global_step)

        self.prediction_comparison = tf.equal(tf.cast(self.labels, tf.int64), tf.argmax(self.readout, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(self.prediction_comparison, tf.float32))

        tf.summary.scalar('loss', tf.reduce_mean(self.loss))
        self.summary = tf.summary.merge_all()

    def print_convolution(self, last):
        if len(last.op.inputs) > 0:
            self.print_convolution(last.op.inputs[0])

        print(last.get_shape(), " - ", last.op.name)
