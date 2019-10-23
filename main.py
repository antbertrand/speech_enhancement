import os
import torch
import torchaudio
from scipy.io import wavfile

def main():
    filename = 'data/timit_test/DR1/FAKS0/SA1'

    assert os.path.isfile(filename + '.wav')
    assert os.path.isfile(filename + '.WAV')

    data, fs = torchaudio.load_wav(filename + '.WAV')

    print(data)
    print(type(data))
    print(data.shape)
    print(fs)
    print(' ')

    data, fs = torchaudio.load_wav(filename + '.wav')

    print(data)
    print(type(data))
    print(data.shape)
    print(fs)
    print(' ')

    fs, data = wavfile.read(filename + '.wav')

    print(data)
    print(type(data))
    print(data.shape)
    print(fs)



if __name__ == '__main__':
    main()  