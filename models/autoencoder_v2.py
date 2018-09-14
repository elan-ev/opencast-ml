import tensorflow as tf
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
import random


class AutoEncoder:
    """ Second version of an autoencoder-architecture.
        The bottleneck-layer breaks down the 512x512x1-Input to a 1024-feature-vector,
        that is used to recreate the input-image.
        As a loss-function the L1-error is used, because it seems to reduce artifacts (https://arxiv.org/abs/1511.08861).
        The readout-layer maps the bottleneck-layer to a 2-features-vector, which stand for "noise"/"no noise".
        Encoder and Readout-Layer are trained seperately.
        After each training iteration in the encoder, images are created that compare the decoder-output to a
        (randomly chosen) input image, to check how good the recreated image looks like.
        """
    def __init__(self, learning_rate=1e-4):
        self.inputs = tf.placeholder(tf.float32, [None, 512, 512, 1])
        self.readout_labels = tf.placeholder(tf.int32, [None])

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # ---------------------------
        # ------- ENCODER -----------
        # ---------------------------
        net = tf.layers.conv2d(self.inputs, 32, 2)
        net = tf.layers.max_pooling2d(net, [3, 3], 2)
        net = tf.layers.conv2d(net, 64, 2)
        net = tf.layers.conv2d(net, 64, 2)
        net = tf.layers.conv2d(net, 64, 2)
        net = tf.layers.max_pooling2d(net, [3, 3], 2)
        net = tf.layers.conv2d(net, 64, 2)
        net = tf.layers.conv2d(net, 64, 2)
        net = tf.layers.conv2d(net, 64, 2)
        net = tf.layers.max_pooling2d(net, [3, 3], 2)
        net = tf.layers.conv2d(net, 128, 2)
        net = tf.layers.conv2d(net, 128, 2)
        net = tf.layers.conv2d(net, 128, 2)
        net = tf.layers.max_pooling2d(net, [3, 3], 2)
        net = tf.layers.conv2d(net, 64, 2)
        net = tf.layers.conv2d(net, 64, 2)
        net = tf.layers.conv2d(net, 64, 2)
        net = tf.layers.conv2d(net, 64, 2)
        net = tf.layers.max_pooling2d(net, 3, 2)
        net = tf.layers.conv2d(net, 64, 2)
        net = tf.layers.conv2d(net, 64, 2)
        net = tf.layers.conv2d(net, 64, 2)
        net = tf.layers.conv2d(net, 64, 2)
        net = tf.layers.conv2d(net, 64, 2)
        net = tf.layers.conv2d(net, 128, 2)
        net = tf.layers.conv2d(net, 128, 2)
        net = tf.layers.conv2d(net, 256, 2)
        net = tf.layers.conv2d(net, 512, 2)
        net = tf.layers.conv2d(net, 1024, 2)

        # -------------------------------------
        # ------------- READOUT ---------------
        # -------------------------------------
        shape = net.get_shape()
        shape = [tf.shape(net)[0], shape[1] * shape[2] * shape[3]]

        # dropout_layer = tf.layers.dropout(tf.reshape(net, shape), rate=0.2, training=False)
        # self.readout = tf.layers.dense(dropout_layer, 2, name='readout')
        self.readout = tf.layers.dense(tf.reshape(net, shape), 2, name='readout')

        with tf.variable_scope('readout', reuse=True):
            readout_weights = tf.get_variable('kernel')
            readout_biases = tf.get_variable('bias')

        self.loss_readout = tf.losses.sparse_softmax_cross_entropy(labels=self.readout_labels, logits=self.readout)
        self.optimize_readout = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_readout,
                                                                                             global_step=self.global_step,
                                                                                             var_list=[
                                                                                                 readout_weights,
                                                                                                 readout_biases])

        correct_prediction = tf.equal(tf.argmax(self.readout, axis=1, output_type=tf.int32), self.readout_labels)
        self.readout_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # -------------------------------------
        # -------------------------------------
        # -------------------------------------


        # ---------------------------
        # ------- DECODER -----------
        # ---------------------------
        net = tf.layers.conv2d_transpose(net, 512, 9, 2, 'same')
        net = tf.layers.conv2d_transpose(net, 256, 9, 2, 'same')
        net = tf.layers.conv2d_transpose(net, 128, 9, 2, 'same')
        net = tf.layers.conv2d_transpose(net, 64, 9, 2, 'same')
        net = tf.layers.conv2d_transpose(net, 64, 9, 2, 'same')
        net = tf.layers.conv2d_transpose(net, 32, 9, 2, 'same')
        net = tf.layers.conv2d_transpose(net, 32, 9, 2, 'same')
        net = tf.layers.conv2d_transpose(net, 16, 9, 2, 'same')
        net = tf.layers.conv2d_transpose(net, 8, 9, 2, 'same')
        net = tf.layers.conv2d_transpose(net, 1, 9, 1, 'same')

        self.outputs = net

        # print the shapes of each layer
        self.print_convolution(self.outputs)

        self.loss = tf.losses.absolute_difference(self.inputs, self.outputs)
        self.optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=self.global_step)


        tf.summary.scalar('loss', tf.reduce_mean(self.loss))
        self.summary = tf.summary.merge_all()

    def print_convolution(self, last):
        """ Prints the structure of the network to the console by looking from the last node backwards.
        :param last:
        """
        if len(last.op.inputs) > 0:
            self.print_convolution(last.op.inputs[0])

        if 'BiasAdd' not in last.op.name:
            print(last.get_shape(), " - ", last.op.name)


def load_data(batch_size=32):
    """ Loading the spectograms randomly
    :param batch_size:
    :return: a generator which generates the input-image-batches as numpy-arrays
    """
    path = 'D:\\noise\\spectograms\\unsupervised'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    random.shuffle(files)

    steps = len(files) // batch_size

    for i in range(0, len(files), batch_size):
        to = i + batch_size
        if to > len(files):
            to = len(files)
        yield np.array([np.array(Image.open(join(path, f))).reshape(512, 512, 1) for f in files[i:to]]), steps


def load_readout_data(batch_size=32):
    base_dir = "D:\\noise\\"

    # this is used to balance the amount of 0- and 1-Labels.
    # This just works, because the first 344 labels in our audio signal seem to be balanced
    number_of_labels = 344

    with open(base_dir + "tagged_5.4584s.txt") as f:
        tags = [int(t[0]) for t in f.readlines()[0:number_of_labels]]
        # print("zeros:", tags.count(0), " , ones:", tags.count(1))

    mypath = base_dir + "spectograms\\5.4584s"
    images = [f for f in listdir(mypath) if isfile(join(mypath, f))][0:number_of_labels]
    images.sort()

    mixed = list(zip(tags, images))
    random.shuffle(mixed)

    for i in range(0, len(mixed), batch_size):
        to = i + batch_size
        if to >= len(mixed):
            to = len(mixed) - 1

        yield zip(*mixed[i:to])


def test_net(sess, model, prefix='after', amount=5):
    """ Create input-output-image-pairs to check how good the network performes
    :param sess:
    :param model: The AutoEncoder-Class
    :param prefix: Filename-Prefix
    :param amount: how much randomly chosen images should be created
    """
    data, _ = next(load_data(batch_size=amount))
    test_images = sess.run([model.outputs], {model.inputs: data})[0]

    for cnt in range(len(data)):
        img = Image.fromarray(np.uint8(np.array(data[cnt]).reshape([512, 512])), 'L')
        img.save("D:\\noise\\results_v2\\" + prefix + "_" + str(cnt) + "_original.png")
        img = Image.fromarray(np.uint8(np.array(test_images[cnt]).reshape([512, 512])), 'L')
        img.save("D:\\noise\\results_v2\\" + prefix + "_" + str(cnt) + "_output.png")
        cnt += 1


def train_encoder():
    ao = AutoEncoder(learning_rate=1e-3)

    epochs = 100

    saver = tf.train.Saver(max_to_keep=epochs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        test_net(sess, ao, 'before')

        for epoch in range(epochs):
            i = 0
            for data, steps in load_data(batch_size=32):
                _, _loss = sess.run([ao.optimize, ao.loss], {ao.inputs: data})
                print(_loss, '(' + str(i) + ' of ' + str(steps) + ') [Epoch: ' + str(epoch) + ']')
                i += 1

            saver.save(sess, 'ao_checkpoints_v2\\autoencoder', global_step=ao.global_step)

            test_net(sess, ao, prefix='epoch_' + str(epoch))


def train_readout():
    tf.reset_default_graph()

    ao = AutoEncoder(learning_rate=1e-3)

    epochs = 50

    saver = tf.train.Saver(max_to_keep=epochs)

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('ao_checkpoints_v2'))
        sess.run(tf.local_variables_initializer())

        for epoch in range(epochs):
            losses = []
            accuracies = []

            for labels, data in load_readout_data(batch_size=50):
                data = np.array([np.array(Image.open(join("D:\\noise\\spectograms\\5.4584s", f))).reshape(512, 512, 1) for f in data])

                training_data, training_labels = data[0:40], labels[0:40]
                test_data, test_labels = data[40:-1], labels[40:-1]

                _, _loss = sess.run([ao.optimize_readout, ao.loss_readout], {ao.inputs: training_data, ao.readout_labels: training_labels})
                losses.append(_loss)

                _accuracy = sess.run([ao.readout_accuracy], {ao.inputs: test_data, ao.readout_labels: test_labels})
                accuracies.append(_accuracy)

            print('loss:',  np.mean(losses), 'accuracy:', np.mean(accuracies), '[Epoch:', epoch, ']')

            saver.save(sess, 'ao_checkpoints_with_readout_v2\\autoencoder_with_readout', global_step=ao.global_step)


def load_readout_checkpoint(sess):
    ao = AutoEncoder()
    saver = tf.train.Saver()

    saver.restore(sess, tf.train.latest_checkpoint('D:\\noise\\src\\models\\ao_checkpoints_with_readout_v2'))
    sess.run(tf.local_variables_initializer())

    return ao.inputs, ao.readout

if __name__ == '__main__':
    train_encoder()
    train_readout()

