from PyQt5.QtWidgets import (QWidget, QSlider,
                             QLabel, QApplication, QPushButton)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

from create_spectogram import audio_to_complete_spectogram, audio_to_complete_diff_spectogram

from pydub import AudioSegment

import pyaudio
from pydub.utils import make_chunks

import threading

from pylab import *

from models.autoencoder_v2 import load_readout_checkpoint

import tensorflow as tf

import time

class PlayerUI(QWidget):
    """ The Audio-Player to test your networks.
    Plays the audio stream and shows the predictions by a green/red label and its
    certainty at that exact point.
    """
    def __init__(self, input_image, audio_segment, prediction):
        super().__init__()

        self.playing = False
        self.chunks = None
        self.stream = None
        self.pyAudio = None
        self.current_index = 0

        self.color_green = "QLabel { background-color: green }"
        self.color_red = "QLabel { background-color: red }"

        self.prediction = prediction

        print('Prediction-Shape:', np.shape(prediction))

        self.initUI(input_image, audio_segment)

    def initUI(self, input_image, audio_segment):
        self.spectogram = input_image

        height, width = input_image.shape
        input_image = np.uint8(input_image).copy()

        print("Image-Shape:", input_image.shape)
        qImg = QImage(input_image.data, width, height, width, QImage.Format_Indexed8)

        width = 3200
        pixmap_image = QPixmap.fromImage(qImg)
        pixmap_image = pixmap_image.scaled(width, height, Qt.IgnoreAspectRatio)
        label_imageDisplay = QLabel(self)
        label_imageDisplay.setPixmap(pixmap_image)

        self.sld = QSlider(Qt.Horizontal, self)
        self.sld.setFocusPolicy(Qt.NoFocus)
        self.sld.setGeometry(0, 520, width, 30)
        self.sld.setRange(0, 10000)
        self.sld.valueChanged[int].connect(self.changeValue)

        self.play_btn = QPushButton(self)
        self.play_btn.setText("Play/Pause")
        self.play_btn.setGeometry(30, height + 140, 100, 50)
        self.play_btn.clicked.connect(self.play_pause)

        self.stop_btn = QPushButton(self)
        self.stop_btn.setText("Stop")
        self.stop_btn.setGeometry(160, height + 140, 00, 50)
        self.stop_btn.clicked.connect(self.stop)

        self.indicator = QLabel(self)
        self.indicator.setAlignment(Qt.AlignCenter)
        self.indicator.setGeometry(width - 120, height + 140, 75, 75)
        self.indicator.setStyleSheet(self.color_red)

        self.audio = audio_segment

        self.setGeometry(500, 300, width, height + 250)
        self.setWindowTitle('Player')

        self.window_open = True

        self.playAt(0)

        self.update_thread = threading.Thread(target=self.update)
        self.update_thread.start()

        self.show()

    def closeEvent(self, event):
        self.window_open = False

    def update(self):
        while self.window_open:
            if self.playing and self.chunks is not None:
                self.stream.write(self.chunks[self.current_index]._data)
                self.sld.valueChanged[int].disconnect(self.changeValue)
                self.sld.setSliderPosition(10000.0 / len(self.chunks) * self.current_index)
                self.sld.valueChanged[int].connect(self.changeValue)

                self.predict()

                self.current_index += 1

                if self.current_index >= len(self.chunks):
                    self.current_index = 0
                    self.sld.valueChanged[int].disconnect(self.changeValue)
                    self.sld.setSliderPosition(0)
                    self.sld.valueChanged[int].connect(self.changeValue)
                    self.playAt(0)

            time.sleep(0.02)

    def changeValue(self, value):
        if self.chunks is not None:
            self.current_index = int(value / (10000.0 / len(self.chunks)))

    def play_pause(self):
        if self.playing:
            self.playing = False
        else:
            if self.chunks is None:
                self.playAt(0)
            self.playing = True

    def playAt(self, index):
        seg = self.audio[index:-1]

        self.pyAudio = pyaudio.PyAudio()
        self.stream = self.pyAudio.open(format=self.pyAudio.get_format_from_width(seg.sample_width),
                        channels=seg.channels,
                        rate=seg.frame_rate,
                        output=True)

        self.chunks = make_chunks(seg, 50)


    def stop(self):
        self.stream.stop_stream()
        self.stream.close()

        self.pyAudio.terminate()

        self.playing = False
        self.chunks = None
        self.stream = None
        self.pyAudio = None
        self.current_index = 0

        self.sld.valueChanged[int].disconnect(self.changeValue)
        self.sld.setSliderPosition(0)
        self.sld.valueChanged[int].connect(self.changeValue)

    def predict(self):
        current_window = int((self.spectogram.shape[1] / len(self.chunks) * self.current_index))
        is_noise = self.prediction[current_window][0] == 0
        self.indicator.setText(str(int(self.prediction[current_window][1] * 100)) + ' %')
        if is_noise:
            self.indicator.setStyleSheet(self.color_red)
        else:
            self.indicator.setStyleSheet(self.color_green)


def softmax(x, axis=None):
  e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
  return e_x / np.sum(e_x, axis=axis, keepdims=True)


def predict_stream(spectogram):
    """ Classifies the spectogram at every time-step, based on the trained network. """
    batch_size = 64
    input_width = 512

    with tf.Session() as sess:
        net_inputs, prediction = load_readout_checkpoint(sess)

        ret = []
        batch = []
        for i in range(spectogram.shape[1]):
            if i % 100 == 0:
                print('predicting: ', i, 'of', spectogram.shape[1])

            current_window = i
            if current_window < input_width / 2:
                current_window = input_width / 2
            elif current_window - (input_width/2) + input_width >= spectogram.shape[1]:
                current_window = spectogram.shape[1] - (input_width / 2)

            start = int(current_window - (input_width / 2))
            end = int(start + input_width)

            batch.append(np.reshape(spectogram[:, start:end], [512, input_width, 1]).astype('uint8'))

            if len(batch) == batch_size:
                output = sess.run(prediction, {net_inputs: np.array(batch)})
                softmaxed = softmax(output, 1)
                result = np.argmax(output, axis=1)
                result = [(result[r], softmaxed[r][result[r]]) for r in range(len(result))]
                ret.extend(result)
                batch = []

    return np.array(ret)

def read_data(audio_file, rng):
    sound_info = audio_to_complete_spectogram(audio_file, rng)
    audio_segment = AudioSegment.from_wav(audio_file)[rng.start*1000:rng.stop*1000]

    return sound_info, audio_segment

if __name__ == '__main__':
    rng = range(840, 860)
    # rng = range(1000, 1010)
    #spectogram, audio_segment = read_data('D:\\noise\\records\\1ff235e4-01e8-469f-a8af-87395bfd7f0d.wav', rng)
    spectogram, audio_segment = read_data('D:\\noise\\records\\8e0223f9-4357-4fbb-8ede-f81477dec101.wav', rng)
    predictions = predict_stream(spectogram)

    app = QApplication(sys.argv)
    ex = PlayerUI(spectogram, audio_segment, predictions)
    sys.exit(app.exec_())