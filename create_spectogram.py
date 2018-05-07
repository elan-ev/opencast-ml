#!/usr/bin/env python
# coding: utf-8
""" This work is licensed under a Creative Commons Attribution 3.0 Unported License.
    Frank Zalkow, 2012-2013 """

import numpy as np
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt

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


def plotstft(audiopath, binsize=2 ** 10, seconds=10.0):
    samplerate, samples = wav.read(audiopath)

    print("Samplerate:", samplerate)

    current = 0
    step_size = int(samplerate * seconds)
    while current < len(samples):
        s = stft(samples[current:current+step_size], binsize)

        sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
        img = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel

        yield np.transpose(np.transpose(np.transpose(img)))
        current += step_size


def mse(img):
    flat = img.flatten()
    max_error = 255 * len(flat)
    return np.sum(np.square(flat)) / max_error
    # return np.median(np.square(img.flatten()))

if __name__ == '__main__':
    seconds = 0.5

    wav1 = plotstft("D:\\noise\\original_short.wav", seconds=seconds)
    wav2 = plotstft("D:\\noise\\filtered_short.wav", seconds=seconds)

    i = 0
    mses = []
    for part1, part2 in zip(wav1, wav2):
        #img = np.concatenate((part1, part2))
        #img = Image.fromarray(np.uint8(img), 'L')
        #img.save("/home/sebi/audio-tests/spectograms/spectogram_" + str(i) + ".png")

        diff = np.absolute(part1 - part2)
        img = Image.fromarray(np.uint8(diff), 'L')
        img.save("D:\\noise\\spectograms\\spectogram_" + str(i).zfill(5) + "_diff.png")

        #all_in_one = np.concatenate((part1, diff, part2))
        #img = Image.fromarray(np.uint8(all_in_one), 'L')
        #img.save("/home/sebi/audio-tests/spectograms/spectogram_" + str(i) + "_all_in_one.png")

        tmp = mse(diff)
        mses.append(tmp)

        i += 1

        if i > 2000:
            break

        if i == (90 // seconds):
            print(tmp)
            #plt.hist(diff.flatten(), bins=200)
            #plt.show()

        if i == (170 // seconds):
            print(tmp)
            #plt.hist(diff.flatten(), bins=200)
            #plt.show()

        if i % (1000 // seconds) == 0:
            print(str(i * seconds), ":", tmp, "    (low:", np.min(mses), ", high:", np.max(mses), ", avg:", np.mean(mses))
            #plt.hist(mses, bins=200)
            #plt.show()


    mses -= np.min(mses)
    mses /= np.max(mses)
    plt.hist(mses, bins=500)
    plt.show()
