import numpy as np
import os
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram, MelScale
from torchaudio.compliance import kaldi
import numpy as np
import matplotlib.pyplot as plt

from utils import sec_to_hms


##################################################
# Main

def main():
    root_dir = './data/raw'
    csv_raw_train = './train_raw.csv'
    csv_raw_test = './test_raw.csv'
    csv_noise_train = './train_noise.csv'
    csv_noise_test = './test_noise.csv'
    
    snr = 1 # TODO check if in dB or not
    
    noiseDataset = NoiseDataset(csv_noise_train, fs=None)
    # TODO décider si je me sers de NoiseDataset en dehors ou en dedans de CustomDataset
    
    in_raw_tf = noiseDataset.add_noise_snr
    in_raw_tf_kwargs = {"fs": None, "snr":1}

    if not all(os.path.exists(f) for f in (csv_raw_train, csv_raw_test)):
        create_csv(root_dir, train_path=csv_raw_train, test_path=csv_raw_test)

    features_tf = Spectrogram(
        # Size of FFT, creates n_fft // 2 + 1 bins
        n_fft=512,
        # Window size. (Default: n_fft)
        win_length=None,
        # Length of hop between STFT windows. ( Default: win_length // 2)
        hop_length=100,
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

    train_set = CustomDataset(root_dir, csv_raw_train, csv_noise_train,
                              in_raw_tf=in_raw_tf, in_raw_tf_kwargs=in_raw_tf_kwargs,
                              target_raw_tf=None, target_raw_tf_kwargs={},
                              features_tf=features_tf, features_tf_kwargs={},
                              in_feats_tf=None, in_feats_tf_kwargs={},
                              target_feats_tf=None, target_feats_tf_kwargs={})
    test_set = CustomDataset(root_dir, csv_raw_test, csv_noise_test,
                              in_raw_tf=in_raw_tf, in_raw_tf_kwargs=in_raw_tf_kwargs,
                              target_raw_tf=None, target_raw_tf_kwargs={},
                              features_tf=features_tf, features_tf_kwargs={},
                              in_feats_tf=None, in_feats_tf_kwargs={},
                              target_feats_tf=None, target_feats_tf_kwargs={})

    # Plot histograms of lengths
    # _, ax = plt.subplots()
    # train_set.histogram_wav_length(ax, label='Train')
    # test_set.histogram_wav_length(ax, label='Test')
    # ax.legend()
    # plt.show()

    # Get an item and plot it
    x, y = train_set[56]
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(x, label='input')  # ; ax1.legend()
    ax2.imshow(y, label='ground truth')  # ; ax2.legend()
    plt.show()


##################################################
# Classes

class CustomDataset(Dataset):
    """TIMIT dataset."""

    def __init__(self, root_dir, csv_raw, fs=None,
                 in_raw_tf=None, in_raw_tf_kwargs={},
                 target_raw_tf=None, target_raw_tf_kwargs={},
                 features_tf=None, features_tf_kwargs={},
                 in_feats_tf=None, in_feats_tf_kwargs={},
                 target_feats_tf=None, target_feats_tf_kwargs={}):

        self.root_dir = root_dir
        self.csv_raw = csv_raw
        self.fs = fs
        self.in_raw_tf = in_raw_tf
        self.in_raw_tf_kwargs = in_raw_tf_kwargs
        self.target_raw_tf = target_raw_tf
        self.target_raw_tf_kwargs = target_raw_tf_kwargs
        self.features_tf = features_tf
        self.features_tf_kwargs = features_tf_kwargs
        self.in_feats_tf = in_feats_tf
        self.in_feats_tf_kwargs = in_feats_tf_kwargs
        self.target_feats_tf = target_feats_tf
        self.target_feats_tf_kwargs = target_feats_tf_kwargs

        with open(csv_raw, 'r') as f:
            self.raw_paths = f.read().split('\n')

    def __len__(self):
        return len(self.raw_paths)

    def __getitem__(self, index):

        raw_target, fs = torchaudio.load(self.raw_paths[index])  # Ground truth

        # Transform the raw ground truth
        if self.target_raw_tf is not None:
            self.target_raw_tf(raw_target, **self.target_raw_tf_kwargs)

        # Add noise to the raw ground truth to create the raw input
        raw_in = raw_target
        if self.in_raw_tf is not None:
            raw_in = self.in_raw_tf(raw_in, **self.in_raw_tf_kwargs)

        # Exctract features
        if self.features_tf is not None:
            x, y = (self.features_tf(a, **self.features_tf_kwargs) for a in (raw_in, raw_target))
        else:
            x, y = raw_in, raw_target  # No features extraction

        # Transform input features
        if self.in_feats_tf is not None:
            x = self.in_feats_tf(x, **self.in_feats_tf_kwargs)
        # Transform target features
        if self.target_feats_tf is not None:
            y = self.target_feats_tf(y, **self.target_feats_tf_kwargs)
            
        # TODO look for STFT output, to handle sliding windows.

        return x.squeeze(), y.squeeze()

    def histogram_wav_length(self, ax=None, label=None):

        print('Computing histogram of lengths.')
        rates = np.zeros(len(self))
        n_samples = np.zeros(len(self))
        for i, wav_path in enumerate(self.raw_paths):
            if not(i % 100):
                print('#{:03d}/{:d}'.format(i+1, len(self)), end='\r')
            si, _ = torchaudio.info(wav_path)
            rates[i], n_samples[i] = si.rate, si.length

        rate = rates[0]
        if not all(r == rate for r in rates):
            print('WARNING : not all rates are the same')
            return

        if ax is None:
            _, ax = plt.subplots()
        n, _, p = ax.hist(n_samples, bins=50)
        ax.set_title('Histogramme des durées')
        xticks = ax.get_xticks()
        ax.set_xticklabels(["{:.0f}\n{:.2f}".format(t, t/rate)
                            for t in xticks])
        ax.set_xlabel("Nombre d'échantillons [] | Durée [s]")

        if label is not None:
            p[0].set_label('{} ({:d} samples | {:d}h{:02d}min{:02.0f}s)'.format(
                label, int(sum(n_samples)), *sec_to_hms(sum(n_samples)/rate)))


class NoiseDataset(Dataset):

    def __init__(self, csv_noise, fs=None):
        """ If fs is not None, then the noise file will be resampled to fs.

        Arguments:
            csv_noise {[type]} -- csv file containing path to each noisy sound
                                  file

        Keyword Arguments:
            fs {int} -- wanted output sampling rate (default: {None})
        """
        self.csv_noise = csv_noise
        self.fs = fs

        with open(csv_noise, 'r') as f:
            self.noise_paths = f.read().split('\n')

        self.nb_samples_cumsum = self.__init_nb_samples_cumsum()

        # for random noise generation
        import time
        torch.random.manual_seed(int(1e9*time.time()))

    def __init_nb_samples_cumsum(self):
        """ Returns a list of the cumsum of the lengths of each noise file in 
        the dataset, in number of samples. The purpose is to better randomize
        noise generation.

        Returns:
            torch.tensor(dtype=torch.long) -- Length of each noise file in the
            dataset, in number of samples. Size torch.Size([len(self)]).
        """
        rates = np.zeros(len(self))
        n_samples = np.zeros(len(self))
        for i, noise_path in enumerate(self.noise_paths):
            # 2 times faster than wave.getnframes
            si, _ = torchaudio.info(noise_path)
            rates[i], n_samples[i] = si.rate, si.length

        rate = rates[0]
        if not all(r == rate for r in rates):
            print('WARNING : not all rates are the same')
            return

        n_samples = torch.tensor(n_samples, dtype=torch.long)

        return n_samples.cumsum(0)

    def __len__(self):
        return len(self.noise_paths)

    def __getitem__(self, index):
        """Returns an audio noise file as a tensor. Index is according to the
        given csv file (`self.csv_noise`).

        Arguments:
            index {int} -- File index, accordinf to the given csv file.

        Returns:
            torch.tensor(dtype=torch.double) -- shape torch.Size([length of the sound file in samples])
        """
        noise, fs = torchaudio.load(self.noise_paths[index])
        noise = noise.squeeze()

        # resample to the dataset wanted fs `self.fs`
        if (self.fs is not None) and (self.fs != fs):
            kaldi.resample_waveform(noise, fs, self.fs)

        return torch.tensor(noise, dtype=torch.double)

    def gen_noise(self, noise_len, fs=None):
        """Randomly generate noise by selecting a sequence from the noise
        dataset

        Arguments:
            noise_len {int} -- length of the wanted sequence in number of
                               samples

        Returns:
            torch.tensor(dtype=torch.double) -- sequence of noise.
                                                shape torch.Size([noise_len])
        """
        
        if fs is not None:
            fs_old = self.fs
            self.fs = fs

        # generate random index
        while True:  # do ... while
            index = torch.randint(low=0, high=self.nb_samples_cumsum[-1],
                                  size=(1,))
            # convert index into sound_index + sample_index
            sound_index = np.searchsorted(self.nb_samples_cumsum, index)
            sample_index = index - self.nb_samples_cumsum[sound_index]

            # check if there are enough samples at the right of the sound
            if (self.nb_samples_cumsum[sound_index] - index) > noise_len:
                break # TODO fix it

        # TODO read with open + struct instead of loading the whole file
        noise = self[sound_index][sample_index:sample_index+noise_len]
        if fs is not None:
            self.fs = fs_old
        
        return noise
    
    def add_noise_snr(self, sig, *, fs=None, snr):
        # TODO handle size better than by squeezing
        noise = self.gen_noise(noise_len=len(sig.squeeze()), fs=fs)
        return add_noise_snr(sig, noise, snr)


##################################################
# Functions    

def add_noise_snr(sig, noise, snr):
    # TODO handle snr
    return sig + noise


def create_csv(root_dir, train_path='./train_raw.csv', test_path='./test_raw.csv'):
    """Create a csv file for TIMIT corpus. 

    Arguments:
        root_dir {str} -- root directory from which create_csv will search for raw data

    Keyword Arguments:
        train_path {str} -- output csv filepath for train data (default: {'./train_raw.csv'})
        test_path {str} -- output csv filepath for test data (default: {'./test_raw.csv'})

    Returns:
        tuple(str, str) -- Tuple (train_path, test_path) with created csv filepaths
    """

    wav_paths_train = []
    wav_paths_test = []

    for root, _, files in os.walk(root_dir):

        if ('train' in root) and (files is not None) and any(f.endswith('.wav') for f in files):
            wav_files = [f for f in files if f.endswith('.wav')]
            wav_paths_train += [os.path.join(root, f) for f in wav_files]

        if ('test' in root) and (files is not None) and any(f.endswith('.wav') for f in files):
            wav_files = [f for f in files if f.endswith('.wav')]
            wav_paths_test += [os.path.join(root, f) for f in wav_files]

    with open(train_path, 'w') as f:
        f.write('\n'.join(wav_paths_train))
    with open(test_path, 'w') as f:
        f.write('\n'.join(wav_paths_test))

    return train_path, test_path


def spectrogram():
    pass

##################################################
# Main


if __name__ == '__main__':
    main()