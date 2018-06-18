import tensorflow as tf
import distinguish
import numpy as np
from create_spectogram import audio_to_complete_spectogram

def create_tag_file():
    lines = []
    step = 5000
    h = 0
    m = 0
    s = 0
    ms = 0
    for i in range(0, 120 * 60 * 1000, step):
        ms += step
        if ms >= 1000:
            ms -= step
            s += int(step/1000)
        if s >= 60:
            s -= 60
            m += 1

        if m >= 60:
            m -= 60
            h += 1

        lines.append("0-" + '{num:02d}'.format(num=h) + ":" + '{num:02d}'.format(num=m) + ":" + '{num:02d}'.format(num=s) + ":" + '{num:02d}'.format(num=ms))

    thefile = open('tagged_empty.txt', 'w')
    for item in lines:
      thefile.write("%s\n" % item)


def change_tag_interval():
    steps = int(10 * 0.5)

    input_file = open('tagged.txt', 'r')
    out = open('tagged_new.txt', 'w')

    tagged = [x.strip() for x in input_file.readlines()]
    new_lines = []
    for i in range(0, len(tagged), steps):
        tag = 0
        if 1 in [int(t[0]) for t in tagged[i:i+steps]]:
            tag = 1

        new_lines.append(tag)

    for item in new_lines:
      out.write("%s\n" % item)


def check_uncertainty_of_net():
    spectogram = audio_to_complete_spectogram('D:\\noise\\records\\1ff235e4-01e8-469f-a8af-87395bfd7f0d_cut.wav')
    batch_size = 32
    input_width = 469

    with tf.Session() as sess:
        readout, inputs, dropout_rate = distinguish.load_checkpoint(sess)

        batch = []
        ret = []
        for i in range(spectogram.shape[1]):
            current_window = i
            if current_window < input_width / 2:
                current_window = input_width / 2
            elif current_window - (input_width / 2) + input_width >= spectogram.shape[1]:
                current_window = spectogram.shape[1] - (input_width / 2)

            start = int(current_window - (input_width / 2))
            end = int(start + input_width)

            batch.append(np.reshape(spectogram[:, start:end], [513, input_width, 1]).astype('uint8'))

            if len(batch) == batch_size:
                output = sess.run(readout, {inputs: np.array(batch), dropout_rate: 0})
                output = np.argmax(output, axis=1)

                ret.extend(output)
                batch = []

    return batch

print(check_uncertainty_of_net())