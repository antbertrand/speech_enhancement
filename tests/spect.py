import matplotlib.pyplot as plt
import torch
from torchaudio.transforms import Spectrogram
from scipy.signal import spectrogram


def main():
    filename = './tests/SA1.wav'
    wavform, fs = read_wav(filename)
    start = 10000
    wavform = wavform[:, start:start+4096]
    print(wavform.shape)

    n_fft = 256
    hop_length = 128

    features_tf = Spectrogram(
        # Size of FFT, creates n_fft // 2 + 1 bins
        n_fft=n_fft,
        # Window size. (Default: n_fft)
        win_length=None,
        # Length of hop between STFT windows. ( Default: win_length // 2)
        hop_length=hop_length,
        # Two sided padding of signal. (Default: 0)
        pad=0,
        # A fn to create a window tensor that is applied/multiplied to each frame/window. (Default: torch.hann_window)
        window_fn=torch.hann_window,
        # Exponent for the magnitude spectrogram, (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: 2)
        power=2,
        # Whether to normalize by magnitude after stft. (Default: False)
        normalized=False,
        # Arguments for window function. (Default: None)
        wkwargs=None
    )

    spect = features_tf(wavform)

    assert spect.shape[2] == wavform.shape[1] // hop_length + 1

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.plot(wavform.t())
    ax2.imshow(spect.squeeze())
    plt.show()

    return


if __name__ == '__main__':
    # Imports from parent directories (https://stackoverflow.com/a/27876800/10076676)
    import sys
    from os import path
    sys.path.insert(1, path.dirname(path.dirname(path.abspath(__file__))))
    from utils.wavutils import read_wav

    main()
