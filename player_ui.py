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

from distinguish import load_checkpoint

import tensorflow as tf

import time

class PlayerUI(QWidget):

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

        self.initUI(input_image, audio_segment)

    def initUI(self, input_image, audio_segment):
        self.spectogram = input_image

        height, width = input_image.shape
        input_image = np.uint8(input_image).copy()

        print("Shape:", input_image.shape)
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
        self.indicator.setGeometry(width - 120, height + 140, 75, 75)
        self.indicator.setStyleSheet(self.color_red)

        self.audio = audio_segment

        self.setGeometry(500, 300, width, height + 250)
        self.setWindowTitle('Player')

        self.window_open = True

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

            time.sleep(0.01)

    def changeValue(self, value):
        if self.chunks is not None:
            self.current_index = int(value / (10000.0 / len(self.chunks)))

    def play_pause(self):
        if self.playing:
            self.playing = False
        else:
            self.playing = True
            if self.chunks is None:
                self.playAt(0)

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
        current_window = int(self.spectogram.shape[1] / len(self.chunks) * self.current_index)
        is_noise = self.prediction[current_window] == 0
        if is_noise:
            self.indicator.setStyleSheet(self.color_red)
        else:
            self.indicator.setStyleSheet(self.color_green)

def predict_stream(spectogram):
    batch_size = 72

    with tf.Session() as sess:
        prediction, net_inputs, dropout_rate = load_checkpoint(sess)

        ret = []
        batch = []
        for i in range(spectogram.shape[1]):
            current_window = i
            if current_window < 47 / 2:
                current_window = 47 / 2
            elif current_window - (47/2) + 47 >= spectogram.shape[1]:
                current_window = spectogram.shape[1] - (47 / 2)

            start = int(current_window - (47 / 2))
            end = int(start + 47)

            batch.append(np.reshape(spectogram[:, start:end], [513, 47, 1]).astype('uint8'))

            if len(batch) == batch_size:
                output = sess.run(prediction, {net_inputs: np.array(batch), dropout_rate: 0})
                output = np.argmax(output, axis=1)
                ret.extend(output)
                batch = []

    return ret

def read_data(audio_file):
    sound_info = audio_to_complete_spectogram(audio_file)
    audio_segment = AudioSegment.from_wav(audio_file)

    return sound_info, audio_segment

if __name__ == '__main__':
    spectogram, audio_segment = read_data('D:\\noise\\short.wav')
    predictions = predict_stream(audio_to_complete_diff_spectogram('D:\\noise\\short.wav', 'D:\\noise\\short_filtered.wav'))
    app = QApplication(sys.argv)
    ex = PlayerUI(spectogram, audio_segment, predictions)
    sys.exit(app.exec_())