import os
import torch
import torchaudio
from scipy.io import wavfile

def main():
    filepath = 'data/timit_test/DR1/FAKS0/SA1.WAV'

    assert os.path.isfile(filepath)

    data, fs = torchaudio.load_wav(filepath)

    print(data)
    print(type(data))
    print(data.shape)
    print(fs)


    # fs, data = wavfile.read(filepath)

    # print(fs)
    # print(data)



if __name__ == '__main__':
    main()  