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
                             QLabel, QApplication)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import sys

from create_spectogram import plotstft
import numpy as np
from PIL import Image

from pydub import AudioSegment

import wave
from pylab import *

class PlayerUI(QWidget):

    def __init__(self, input_image, audio_segment):
        super().__init__()

        self.initUI(input_image, audio_segment)

    def initUI(self, input_image, audio_segment):
        height, width = input_image.shape[1], input_image.shape[0]
        input_image = np.require(input_image, np.uint8, 'C')
        print(input_image.shape)
        qImg = QImage(input_image, width, height, QImage.Format_Grayscale8)
        pixmap01 = QPixmap.fromImage(qImg)
        pixmap_image = QPixmap(pixmap01)
        label_imageDisplay = QLabel(self)
        label_imageDisplay.setPixmap(pixmap_image)
        label_imageDisplay.setGeometry(0, 0, width, height)

        sld = QSlider(Qt.Horizontal, self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setGeometry(12, 400, 2000, 30)
        sld.valueChanged[int].connect(self.changeValue)

        self.setGeometry(500, 300, 2024, 500)
        self.setWindowTitle('Player')
        self.show()

    def changeValue(self, value):
        print(value)


def read_data(audio_file):
    sound_info = next(plotstft(audio_file, seconds=10))
    audio_segment = AudioSegment.from_wav(audio_file)

    return sound_info, audio_segment

if __name__ == '__main__':
    spectogram, audio_segment = read_data('D:\\noise\\original_short.wav')
    app = QApplication(sys.argv)
    ex = PlayerUI(spectogram, audio_segment)
    sys.exit(app.exec_())