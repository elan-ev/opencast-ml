from pydub import AudioSegment
from pydub.utils import make_chunks
import pyaudio


if __name__ == '__main__':
    seg = AudioSegment.from_wav('D:\\noise\\original_short.wav')
    start = 3 * 60 * 1000
    seg = seg[start:-1]

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(seg.sample_width),
                    channels=seg.channels,
                    rate=seg.frame_rate,
                    output=True)

    # break audio into half-second chunks (to allows keyboard interrupts)
    for chunk in make_chunks(seg, 500):
        stream.write(chunk._data)

    stream.stop_stream()
    stream.close()

    p.terminate()