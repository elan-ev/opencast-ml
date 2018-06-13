import tensorflow as tf
from os import listdir
from os.path import isfile, join
from random import shuffle
import numpy as np
from PIL import Image

base_dir = "D:\\noise\\"
number_of_labels = 1127
batch_size = 16
learning_rate = 1e-3

def load_data():
    with open(base_dir + "records\\1ff235e4-01e8-469f-a8af-87395bfd7f0d_cut.txt") as f:
        tags = [int(t[0]) for t in f.readlines()[0:number_of_labels]]
        print("zeros:", tags.count(0), " , ones:", tags.count(1))

    mypath = base_dir + "spectograms\\5s"
    images = [f for f in listdir(mypath) if isfile(join(mypath, f))][0:number_of_labels]
    images.sort()

    mixed = list(zip(tags, images))
    shuffle(mixed)
    return mixed

def init_data():
    data = load_data()
    split_index = int(0.7 * len(data))
    training_data = data[0:split_index]
    test_data = data[split_index:-1]
    return test_data, training_data

def create_batch(data):
    for i in range(len(data) // batch_size):
        on = i * batch_size
        off = on + batch_size
        yield data[on:off]


inputs = tf.placeholder(tf.float32, [None, 513, 469, 1])
labels = tf.placeholder(tf.int32, [None])
dropout_rate = tf.placeholder(tf.float32)


conv_0_1 = tf.layers.conv2d(inputs, 32, 3)
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
print(shape)
shape = [tf.shape(conv_5_2)[0], shape[1] * shape[2] * shape[3]]

dropout = tf.layers.dropout(tf.reshape(conv_5_2, shape), rate=dropout_rate)
readout_0 = tf.layers.dense(dropout, 1024)
dropout_2 = tf.layers.dropout(readout_0, rate=dropout_rate)
readout = tf.layers.dense(dropout_2, 2)

loss = tf.losses.sparse_softmax_cross_entropy(labels, readout)
optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction_comparison = tf.equal(tf.cast(labels, tf.int64), tf.argmax(readout, axis=1))
accuracy = tf.reduce_mean(tf.cast(prediction_comparison, tf.float32))

def load_image(file):
    return np.reshape(list(Image.open(base_dir + "spectograms\\5s\\" + file).getdata()), [513, 469, 1]).astype('uint8')


def check_accuracy(sess, test_data):
    accuracies = []
    errors = []
    for batch in create_batch(test_data):
        unzipped = list(zip(*batch))
        tags = unzipped[0]
        images = [load_image(img) for img in unzipped[1]]

        _accuracy, _prediction = sess.run([accuracy, prediction_comparison],
                                          {inputs: images, labels: tags, dropout_rate: 0})

        errors.extend([(unzipped[0][idx], unzipped[1][idx]) for idx, val in enumerate(_prediction) if val == False])

        accuracies.append(_accuracy)

    print('accuracy:', np.mean(accuracies))

    return errors


def train_net(test_data, training_data):
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        check_accuracy(sess, test_data)

        for i in range(150):
            print("Epoch:", str(i))

            shuffle(training_data)

            losses = []
            for batch in create_batch(training_data):
                unzipped = list(zip(*batch))
                tags = unzipped[0]
                images = [load_image(img) for img in unzipped[1]]

                _, _loss = sess.run([optimize, loss], {inputs: images, labels: tags,
                                                       dropout_rate: 0.3})
                losses.append(_loss)

            print('Losses:', losses)
            print('Mean Loss:', np.mean(losses))
            check_accuracy(sess, test_data)

            if i % 10 == 0:
                saver.save(sess, 'checkpoints\\5s')
                print('Saved epoch: ', i)

        errors = check_accuracy(sess, test_data)
        for error in errors:
            print(error)


def load_checkpoint(sess):
    saver = tf.train.Saver()
    saver.restore(sess, 'D:\\noise\\checkpoints\\test')
    return readout, inputs, dropout_rate


if __name__ == '__main__':
    test_data, training_data = init_data()
    train_net(test_data, training_data)

    #with tf.Session() as sess:
    #    load_checkpoint(sess)
    #    test_data, training_data = init_data()
    #    check_accuracy(sess, test_data)
