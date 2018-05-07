import tensorflow as tf
from os import listdir
from os.path import isfile, join
from random import shuffle
import numpy as np
from PIL import Image

base_dir = "D:\\noise\\"
number_of_labels = 723
batch_size = 72
learning_rate = 3e-4

def load_data():
    with open(base_dir + "tagged.txt") as f:
        tags = [int(t[0]) for t in f.readlines()[0:number_of_labels]]
        print("zeros:", tags.count(0), " , ones:", tags.count(1))

    mypath = base_dir + "spectograms"
    images = [f for f in listdir(mypath) if isfile(join(mypath, f))][0:number_of_labels]
    images.sort()

    mixed = list(zip(tags, images))
    shuffle(mixed)
    return mixed


data = load_data()
split_index = int(0.7 * len(data))
training_data = data[0:split_index]
test_data = data[split_index:-1]

def create_batch(data=training_data):
    for i in range(len(data) // batch_size):
        on = i * batch_size
        off = on + batch_size
        yield data[on:off]


inputs = tf.placeholder(tf.float32, [None, 513, 47, 1])
labels = tf.placeholder(tf.int32, [None])

conv_1_1 = tf.layers.conv2d(inputs, 64, 3)
conv_1_2 = tf.layers.conv2d(conv_1_1, 64, 3)
conv_1_3 = tf.layers.conv2d(conv_1_2, 64, 3)
pool_1 = tf.layers.max_pooling2d(conv_1_3, [3, 2], [3, 1])
conv_2_1 = tf.layers.conv2d(pool_1, 64, 3)
conv_2_2 = tf.layers.conv2d(conv_2_1, 64, 3)
conv_2_3 = tf.layers.conv2d(conv_2_2, 64, 3)
pool_2 = tf.layers.max_pooling2d(conv_2_3, [3, 2], [3, 1])
conv_3_1 = tf.layers.conv2d(pool_2, 128, 3)
conv_3_2 = tf.layers.conv2d(conv_3_1, 128, 3)
conv_3_3 = tf.layers.conv2d(conv_3_2, 128, 3)
pool_3 = tf.layers.max_pooling2d(conv_3_3, [3, 2], [3, 2])
conv_4_1 = tf.layers.conv2d(pool_3, 64, 3)
conv_4_2 = tf.layers.conv2d(conv_4_1, 32, 3)
conv_4_3 = tf.layers.conv2d(conv_4_2, 32, 3)

shape = conv_4_3.get_shape()
shape = [batch_size, int(shape[1] * shape[2] * shape[3])]

dropout = tf.layers.dropout(tf.reshape(conv_4_3, shape), rate=0.4)
readout_0 = tf.layers.dense(dropout, 1024)
dropout_2 = tf.layers.dropout(readout_0, rate=0.4)
readout = tf.layers.dense(dropout_2, 2)

loss = tf.losses.sparse_softmax_cross_entropy(labels, readout)
optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction_comparison = tf.equal(tf.cast(labels, tf.int64), tf.argmax(readout, axis=1))
accuracy = tf.reduce_mean(tf.cast(prediction_comparison, tf.float32))

def load_image(file):
    return np.reshape(list(Image.open(base_dir + "spectograms\\" + file).getdata()), [513, 47, 1]).astype('uint8')


def check_accuracy(sess):
    accuracies = []
    errors = []
    for batch in create_batch(test_data):
        unzipped = list(zip(*batch))
        tags = unzipped[0]
        images = [load_image(img) for img in unzipped[1]]

        _accuracy, _prediction = sess.run([accuracy, prediction_comparison], {inputs: images, labels: tags})

        errors.extend([(unzipped[0][idx], unzipped[1][idx]) for idx, val in enumerate(_prediction) if val == False])

        accuracies.append(_accuracy)

    print('accuracy:', np.mean(accuracies))

    return errors


saver = tf.train.Saver()

def train_net():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        check_accuracy(sess)

        for i in range(150):
            print("Epoch:", str(i))

            shuffle(training_data)

            losses = []
            for batch in create_batch():
                unzipped = list(zip(*batch))
                tags = unzipped[0]
                images = [load_image(img) for img in unzipped[1]]

                _, _loss = sess.run([optimize, loss], {inputs: images, labels: tags})
                losses.append(_loss)

            print('Losses:', losses)
            check_accuracy(sess)

            if i % 50 == 0:
                saver.save(sess, 'D:\\noise\\checkpoints\\')

        errors = check_accuracy(sess)
        for error in errors:
            print(error)

def classify_audio():
    with tf.Session() as sess:
        saver.restore(sess, 'D:\\noise\\checkpoints')
    print("")


classify_audio()

