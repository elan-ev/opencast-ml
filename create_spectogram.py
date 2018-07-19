#!/usr/bin/env python
# coding: utf-8
""" This work is licensed under a Creative Commons Attribution 3.0 Unported License.
    Frank Zalkow, 2012-2013 """

import numpy as np
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from PIL import Image
from os.path import join
import sys

""" short time fourier transform of audio signal """


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    floor = np.floor(frameSize / 2.0)
    zeros = np.zeros(int(floor))
    samples = np.append(zeros, sig)
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


""" scale frequency axis logarithmically """


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins - 1) / max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):int(scale[i + 1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i + 1])])]

    return newspec, freqs


def plotstft(audiopath, binsize=2 ** 10, seconds=5.0, rng=None):
    samplerate, samples = wav.read(audiopath)

    print("Samplerate:", samplerate)

    step_size = int(samplerate * seconds)

    if rng is None:
        rng = range(0, int(len(samples) / step_size))

    for cnt in rng:
        current = cnt * step_size
        s = stft(samples[current:current+step_size], binsize)

        sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
        img = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel

        yield np.transpose(np.transpose(np.transpose(img)))
        current += step_size

def audio_to_complete_spectogram(audiopath, rng=None):
    complete = None
    for part in plotstft(audiopath, rng=rng):
        if complete is None:
            complete = part
        else:
            complete = np.concatenate((complete, part), axis=1)
    return complete


def audio_to_complete_diff_spectogram(audio1, audio2):
    complete = None
    for part1, part2 in zip(plotstft(audio1), plotstft(audio2)):
        if complete is None:
            complete = np.absolute(part1 - part2)
        else:
            complete = np.concatenate((complete, np.absolute(part1 - part2)), axis=1)
    return complete


def mse(img):
    flat = img.flatten()
    max_error = 255 * len(flat)
    return np.sum(np.square(flat)) / max_error
    # return np.median(np.square(img.flatten()))


# Print iterations progress
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    print('\r[%s] %s%s ...%s' % (bar, percents, '%', status), end='\r')

if __name__ == '__main__':
    seconds = 5.4584
    folder = 'D:\\noise\\records\\'

    files = [#'97dae982-3246-431d-a615-86638992f272.wav', '9b742d18-903e-4236-ba35-a7f76790c7d7.wav',
             #'d32f1981-2bb0-4c98-8b16-4e02be6fb256.wav', '5e212b2d-3c6b-4ba0-913f-30c7cd8d8984.wav',
             #'7dcacfcf-2a33-4298-bc52-18afd6437be7.wav', '8e0223f9-4357-4fbb-8ede-f81477dec101.wav',
             #'9f45a952-67d6-4d23-b0ab-a4bcec74fdc1.wav', 'track-1.wav']
            '1ff235e4-01e8-469f-a8af-87395bfd7f0d_cut.wav']

    k = 0
    for file in files:
        f = join(folder, file)

        wav1 = plotstft(f, seconds=seconds)
        wav1 = [arr for arr in wav1]

        complete = np.concatenate(wav1, axis=1)
        print(np.shape(complete))

        counter = 0
        for i in range(0, len(complete[0]), 50):
            part = complete[:, i:i+512]
            img = Image.fromarray(np.uint8(part), 'L').crop((0, 0, 512, 512))
            img.save("D:\\noise\\spectograms\\unsupervised\\file_" + str(k) + "_spectogram_" + str(i).zfill(10) + ".png")

            if counter % 100 == 0:
                progress(i, len(complete[0]), 'File: ' + file)
                counter += 1

        print()

        k += 1
