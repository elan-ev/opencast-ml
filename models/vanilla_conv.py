import tensorflow as tf

class VanillaConv:
    def __init__(self, learning_rate=1e-3):
        self.inputs = tf.placeholder(tf.float32, [None, 513, 469, 1])
        self.labels = tf.placeholder(tf.int32, [None])
        self.dropout_rate = tf.placeholder(tf.float32)

        conv_0_1 = tf.layers.conv2d(self.inputs, 32, 3)
        pool_0 = tf.layers.max_pooling2d(conv_0_1, [3, 3], 2)

        conv_1_1 = tf.layers.conv2d(pool_0, 64, 3)
        conv_1_2 = tf.layers.conv2d(conv_1_1, 64, 3)
        conv_1_3 = tf.layers.conv2d(conv_1_2, 64, 3)
        pool_1 = tf.layers.max_pooling2d(conv_1_3, [3, 3], 2)
        conv_2_1 = tf.layers.conv2d(pool_1, 64, 3)
        conv_2_2 = tf.layers.conv2d(conv_2_1, 64, 3)
        conv_2_3 = tf.layers.conv2d(conv_2_2, 64, 3)
        pool_2 = tf.layers.max_pooling2d(conv_2_3, [3, 3], 2)
        conv_3_1 = tf.layers.conv2d(pool_2, 128, 3)
        conv_3_2 = tf.layers.conv2d(conv_3_1, 128, 3)
        conv_3_3 = tf.layers.conv2d(conv_3_2, 128, 3)
        pool_3 = tf.layers.max_pooling2d(conv_3_3, [3, 3], 2)
        conv_4_1 = tf.layers.conv2d(pool_3, 64, 3)
        conv_4_2 = tf.layers.conv2d(conv_4_1, 32, 3)
        conv_4_3 = tf.layers.conv2d(conv_4_2, 32, 3)
        pool_4 = tf.layers.max_pooling2d(conv_4_3, 3, 2)
        conv_5_1 = tf.layers.conv2d(pool_4, 32, 3)
        conv_5_2 = tf.layers.conv2d(conv_5_1, 32, 3)

        shape = conv_5_2.get_shape()
        shape = [tf.shape(conv_5_2)[0], shape[1] * shape[2] * shape[3]]

        dropout = tf.layers.dropout(tf.reshape(conv_5_2, shape), rate=self.dropout_rate)
        readout_0 = tf.layers.dense(dropout, 1024)
        dropout_2 = tf.layers.dropout(readout_0, rate=self.dropout_rate)
        self.readout = tf.layers.dense(dropout_2, 2)

        # print the shapes of each layer
        self.print_convolution(self.readout)

        self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.readout)
        self.optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        self.prediction_comparison = tf.equal(tf.cast(self.labels, tf.int64), tf.argmax(self.readout, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(self.prediction_comparison, tf.float32))

    def print_convolution(self, last):
        if len(last.op.inputs) > 0:
            self.print_convolution(last.op.inputs[0])

        print(last.get_shape(), " - ", last.op.name)
