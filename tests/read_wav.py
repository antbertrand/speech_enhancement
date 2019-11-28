""" Testing libraries to find the fastest way to read a wav file

Librosa not here because definitely slower.

Conclusion :
- Use `scipy.io.wavfile.read()` when loading the whole file.
- Use `wave` + `np.from_buffer()` when loading a part.
"""
import numpy as np
import wave
import torch
import torchaudio
from scipy.io import wavfile
import timeit


def main():

    # Parameters
    filename = 'SA1.wav'
    offset = 2048
    nframes = 1024

    # Check for consistence
    if True:

        # Loading whole file
        x1 = load1(filename)
        x2 = load2(filename)
        x3 = load3(filename)
        assert torch.all(torch.eq(x1, x2)) and torch.all(torch.eq(x2, x3))

        # Loading a part
        x1 = load1_part(filename, offset, nframes)
        x2 = load2_part(filename, offset, nframes)
        x3 = load3_part(filename, offset, nframes)
        assert torch.all(torch.eq(x1, x2)) and torch.all(torch.eq(x2, x3))

    # Time it
    if True:

        # Loading the whole file
        print('  Loading the whole file.')
        print('#1 wave:')
        print(timeit.timeit("load1('SA1.wav')",
                            setup="from __main__ import load1", number=10000))
        print('\n#2 torchaudio:')
        print(timeit.timeit("load2('SA1.wav')",
                            setup="from __main__ import load2", number=10000))
        print('\n#3 scipy:')
        print(timeit.timeit("load3('SA1.wav')",
                            setup="from __main__ import load3", number=10000))

        # Loading a part
        print('\n\n  Loading a part.')
        print('#1 wave:')
        print(timeit.timeit("load1_part('SA1.wav', 2048, 1024)",
                            setup="from __main__ import load1_part", number=10000))
        print('\n#2 torchaudio:')
        print(timeit.timeit("load2_part('SA1.wav', 2048, 1024)",
                            setup="from __main__ import load2_part", number=10000))
        print('\n#3 scipy:')
        print(timeit.timeit("load3_part('SA1.wav', 2048, 1024)",
                            setup="from __main__ import load3_part", number=10000))


def load1(filename):
    # With wave + numpy.frombuffer
    with wave.open(filename) as f:
        fs = f.getframerate()
        buff = f.readframes(f.getnframes())
    x = torch.tensor(np.frombuffer(buff, np.int16), dtype=torch.double)
    x.unsqueeze_(dim=0)

    return x


def load2(filename):
    # With torchaudio.load_wav
    x, fs = torchaudio.load_wav(filename)
    x = x.to(dtype=torch.double)

    return x


def load3(filename):
    # With scipy.io.wavfile.read
    fs, x = wavfile.read(filename, mmap=False)
    x = torch.tensor(x, dtype=torch.double)
    x.unsqueeze_(dim=0)

    return x


def load1_part(filename, offset, nframes):
    # With wave + numpy.frombuffer
    with wave.open(filename) as f:
        fs = f.getframerate()
        f.setpos(offset)
        buff = f.readframes(nframes)
    x = torch.tensor(np.frombuffer(buff, np.int16), dtype=torch.double)
    x.unsqueeze_(dim=0)

    return x


def load2_part(filename, offset, nframes):
    # With torchaudio.load_wav
    x, fs = torchaudio.load_wav(filename)
    x = x[:, offset:offset+nframes].to(dtype=torch.double)

    return x


def load3_part(filename, offset, nframes):
    # With scipy.io.wavfile.read
    fs, x = wavfile.read(filename, mmap=True)
    x = torch.tensor(x[offset:offset+nframes], dtype=torch.double)
    x.unsqueeze_(dim=0)

    return x


if __name__ == '__main__':
    main()
