#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
ZetCode PyQt5 tutorial

This example shows a QSlider widget.

Author: Jan Bodnar
Website: zetcode.com
Last edited: August 2017
"""

from PyQt5.QtWidgets import (QWidget, QSlider,
                             QLabel, QApplication, QPushButton)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

from create_spectogram import plotstft

from pydub import AudioSegment

import pyaudio
from pydub.utils import make_chunks

from pylab import *

class PlayerUI(QWidget):

    def __init__(self, input_image, audio_segment):
        super().__init__()

        self.playing = False

        self.initUI(input_image, audio_segment)

    def initUI(self, input_image, audio_segment):
        height, width = input_image.shape
        input_image = np.uint8(input_image).copy()

        print(input_image.shape)
        qImg = QImage(input_image.data, width, height, width, QImage.Format_Indexed8)
        pixmap_image = QPixmap.fromImage(qImg)
        label_imageDisplay = QLabel(self)
        label_imageDisplay.setPixmap(pixmap_image)

        sld = QSlider(Qt.Horizontal, self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setGeometry(0, 400, width, 30)
        sld.valueChanged[int].connect(self.changeValue)

        self.play_btn = QPushButton(self)
        self.play_btn.setText("Play/Pause")
        self.play_btn.setGeometry(30, height + 10, 100, 50)
        self.play_btn.clicked.connect(self.play_pause)

        self.audio = audio_segment

        self.setGeometry(500, 300, width, height + 100)
        self.setWindowTitle('Player')
        self.show()

    def changeValue(self, value):
        print(value)

    def play_pause(self):
        if self.playing:
            self.playing = False
        else:
            self.playing = True
            self.playAt(0)

        print(self.playing)

    def playAt(self, index):
        seg = self.audio[index:-1]

        self.pyAudio = pyaudio.PyAudio()
        self.stream = self.pyAudio.open(format=self.pyAudio.get_format_from_width(seg.sample_width),
                        channels=seg.channels,
                        rate=seg.frame_rate,
                        output=True)

        self.chunks = make_chunks(seg, 50)

        # break audio into half-second chunks (to allows keyboard interrupts)
        for chunk in self.chunks:
            self.stream.write(chunk._data)
            break

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()

        self.pyAudio.terminate()


def read_data(audio_file):
    i = 0
    for img in plotstft(audio_file, seconds=30):
        sound_info = img
        i += 1
        if i > 6:
            break

    audio_segment = AudioSegment.from_wav(audio_file)

    return sound_info, audio_segment

if __name__ == '__main__':
    spectogram, audio_segment = read_data('D:\\noise\\original_short.wav')
    app = QApplication(sys.argv)
    ex = PlayerUI(spectogram, audio_segment)
    sys.exit(app.exec_())