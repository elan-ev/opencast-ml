import tensorflow as tf
from os import listdir
from os.path import isfile, join
from random import shuffle
import numpy as np
from PIL import Image

from models.vanilla_conv import VanillaConv

base_dir = "D:\\noise\\"
number_of_labels = 1127
batch_size = 32

model = VanillaConv(learning_rate=1e-3)

# ----------------------------------------------
# These were the first tests on the Vanilla-Convolution.
# Might be helpful so I just leave it here.
# -----------------------------------------------


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


def load_image(file):
    return np.reshape(list(Image.open(base_dir + "spectograms\\5s\\" + file).getdata()), [513, 469, 1]).astype('uint8')


def check_accuracy(sess, test_data, summary_writer):
    accuracies = []
    errors = []
    for batch in create_batch(test_data):
        unzipped = list(zip(*batch))
        tags = unzipped[0]
        images = [load_image(img) for img in unzipped[1]]

        _accuracy, _prediction = sess.run([model.accuracy, model.prediction_comparison],
                                          {model.inputs: images, model.labels: tags, model.dropout_rate: 0})

        errors.extend([(unzipped[0][idx], unzipped[1][idx]) for idx, val in enumerate(_prediction) if val == False])

        accuracies.append(_accuracy)

    summary = tf.Summary()
    summary.value.add(tag="Accuracy", simple_value=np.mean(accuracies))
    summary_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))

    print('accuracy:', np.mean(accuracies))

    return errors


def train_net(test_data, training_data):
    saver = tf.train.Saver()
    test_writer = tf.summary.FileWriter('summary/test')
    train_writer = tf.summary.FileWriter('summary/train')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        check_accuracy(sess, test_data, test_writer)

        for i in range(150):
            print("Epoch:", str(i))

            shuffle(training_data)

            losses = []
            for batch in create_batch(training_data):
                unzipped = list(zip(*batch))
                tags = unzipped[0]
                images = [load_image(img) for img in unzipped[1]]

                _summary, _, _loss = sess.run([model.summary, model.optimize, model.loss], {model.inputs: images, model.labels: tags,
                                                       model.dropout_rate: 0.3})
                losses.append(_loss)

                train_writer.add_summary(_summary, tf.train.global_step(sess, model.global_step))

            print('Losses:', losses)
            print('Mean Loss:', np.mean(losses))
            check_accuracy(sess, test_data, test_writer)

            if i % 10 == 0:
                saver.save(sess, 'checkpoints\\5s', global_step=model.global_step)
                print('Saved epoch: ', i)

        errors = check_accuracy(sess, test_data, test_writer)
        for error in errors:
            print(error)


def load_checkpoint(sess):
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('D:\\noise\\src\\checkpoints\\'))
    return model.readout, model.inputs, model.dropout_rate


if __name__ == '__main__':
    test_data, training_data = init_data()
    train_net(test_data, training_data)

    #with tf.Session() as sess:
        # load_checkpoint(sess)
        #test_data, training_data = init_data()
        #check_accuracy(sess, test_data)
