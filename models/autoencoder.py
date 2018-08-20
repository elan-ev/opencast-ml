import tensorflow as tf
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
import random


class AutoEncoder:
    def __init__(self, learning_rate=1e-4):
        self.inputs = tf.placeholder(tf.float32, [None, 512, 512, 1])
        self.readout_labels = tf.placeholder(tf.int32, [None])

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

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
        net = tf.layers.conv2d(net, 64, 2, padding='same')

        shape = net.get_shape()
        shape = [tf.shape(net)[0], shape[1] * shape[2] * shape[3]]

        self.readout = tf.layers.dense(tf.reshape(net, shape), 2, name='readout')
        with tf.variable_scope('readout', reuse=True):
            readout_weights = tf.get_variable('kernel')
            readout_biases = tf.get_variable('bias')

        net = tf.layers.conv2d_transpose(net, 16, 9, 2, 'same')
        net = tf.layers.conv2d_transpose(net, 64, 9, 2, 'same')
        net = tf.layers.conv2d_transpose(net, 128, 9, 2, 'same')
        net = tf.layers.conv2d_transpose(net, 128, 9, 2, 'same')
        net = tf.layers.conv2d_transpose(net, 64, 9, 2, 'same')
        net = tf.layers.conv2d_transpose(net, 1, 9, 2, 'same')
        net = tf.layers.conv2d_transpose(net, 1, 9, 1, 'same')

        self.outputs = net

        # print the shapes of each layer
        self.print_convolution(self.outputs)

        self.loss = tf.losses.mean_squared_error(self.inputs, self.outputs)
        self.optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=self.global_step)

        self.loss_readout = tf.losses.sparse_softmax_cross_entropy(labels=self.readout_labels, logits=self.readout)
        self.optimize_readout = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_readout,
                                                                                             global_step=self.global_step, var_list=[readout_weights, readout_biases])

        tf.summary.scalar('loss', tf.reduce_mean(self.loss))
        self.summary = tf.summary.merge_all()

    def print_convolution(self, last):
        if len(last.op.inputs) > 0:
            self.print_convolution(last.op.inputs[0])

        print(last.get_shape(), " - ", last.op.name)


def load_data(batch_size=32):
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
    data, _ = next(load_data(batch_size=amount))
    test_images = sess.run([model.outputs], {model.inputs: data})[0]

    for cnt in range(len(data)):
        img = Image.fromarray(np.uint8(np.array(data[cnt]).reshape([512, 512])), 'L')
        img.save("D:\\noise\\results\\" + prefix + "_" + str(cnt) + "_original.png")
        img = Image.fromarray(np.uint8(np.array(test_images[cnt]).reshape([512, 512])), 'L')
        img.save("D:\\noise\\results\\" + prefix + "_" + str(cnt) + "_output.png")
        cnt += 1


def train_encoder():
    ao = AutoEncoder()

    epochs = 100

    saver = tf.train.Saver(max_to_keep=epochs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        test_net(sess, ao, 'before')

        for epoch in range(epochs):
            i = 0
            for data, steps in load_data():
                _, _loss = sess.run([ao.optimize, ao.loss], {ao.inputs: data})
                print(_loss, '(' + str(i) + ' of ' + str(steps) + ') [Epoch: ' + str(epoch) + ']')
                i += 1

            saver.save(sess, 'ao_checkpoints\\autoencoder', global_step=ao.global_step)

            test_net(sess, ao, prefix='epoch_' + str(epoch))


def train_readout():
    tf.reset_default_graph()

    ao = AutoEncoder(learning_rate=3e-2)

    epochs = 500

    saver = tf.train.Saver(max_to_keep=epochs)

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('ao_checkpoints'))
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

            saver.save(sess, 'ao_checkpoints_with_readout\\autoencoder_with_readout', global_step=ao.global_step)


def load_readout_checkpoint(sess):
    ao = AutoEncoder()
    saver = tf.train.Saver()

    saver.restore(sess, tf.train.latest_checkpoint('D:\\noise\\src\\models\\ao_checkpoints_with_readout'))
    sess.run(tf.local_variables_initializer())

    return ao.inputs, ao.readout

if __name__ == '__main__':
    train_encoder()
    train_readout()

