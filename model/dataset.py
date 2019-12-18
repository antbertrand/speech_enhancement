import os

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io.wavfile as wave

import torch
import torchaudio
from torch.utils.data import Dataset
from librosa.core import stft as librosa_stft
from torchvision.transforms.functional import normalize

if True:  # Not to break code order with autoformatter
    # Needed here, and not under ifmain, because @time decorator is imported
    import sys
    from os import path
    sys.path.insert(1, path.dirname(path.dirname(path.abspath(__file__))))
    from utils.utils import sec_to_hms
    from utils.wavutils import read_wav

##################################################
# Main


def main():
    root_dir = './data/raw'
    csv_raw_train = './data/train_raw.csv'
    csv_raw_test = './data/test_raw.csv'
    csv_noise_train = './data/train_noise.csv'
    csv_noise_test = './data/test_noise.csv'

    snr = 0  # TODO check if in dB or not

    noiseDataset = NoiseDataset(csv_noise_train, fs=None)

    in_raw_tf = noiseDataset.add_noise_snr
    in_raw_tf_kwargs = {"fs": None, "snr": 1}

    # check if csv files are done
    if not all(os.path.exists(f) for f in (csv_raw_train, csv_raw_test)):
        create_csv(root_dir, train_path=csv_raw_train, test_path=csv_raw_test)

    features_tf = stft
    n_fft = 256
    hop_length = n_fft // 2

    features_tf_kwargs = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": None,
        "window": torch.hann_window(n_fft).numpy(),  # 'hann',
        "center": True,
        "dtype": np.complex64,
        "pad_mode": 'reflect'  # 'reflect'
    }

    train_set = CustomDataset(root_dir, csv_raw_train, csv_noise_train,
                              in_raw_tf=in_raw_tf, in_raw_tf_kwargs=in_raw_tf_kwargs,
                              target_raw_tf=None, target_raw_tf_kwargs={},
                              features_tf=features_tf, features_tf_kwargs=features_tf_kwargs,
                              in_feats_tf=None, in_feats_tf_kwargs={},
                              target_feats_tf=None, target_feats_tf_kwargs={})
    # test_set = CustomDataset(root_dir, csv_raw_test, csv_noise_test,
    #                          in_raw_tf=in_raw_tf, in_raw_tf_kwargs=in_raw_tf_kwargs,
    #                          target_raw_tf=None, target_raw_tf_kwargs={},
    #                          features_tf=features_tf, features_tf_kwargs={},
    #                          in_feats_tf=None, in_feats_tf_kwargs={},
    #                          target_feats_tf=None, target_feats_tf_kwargs={})

    # Plot histograms of lengths
    # _, ax = plt.subplots()
    # train_set.histogram_wav_length(ax, label='Train')
    # test_set.histogram_wav_length(ax, label='Test')
    # ax.legend()
    # plt.show()

    # Get an item and plot it
    print('Taille dataset', len(train_set))
    x, y = train_set[56]

    print(x.shape)

    #wave.write('./tests/test1.wav', 16000, x)
    #torchaudio.save('./tests/test1.wav', y, 16000)
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    '''
    ax1.imshow(x)
    ax1.set_title('input : {:d}, {:d}'.format(*x.shape))
    ax2.imshow(y)
    ax2.set_title('ground truth : {:d}, {:d}'.format(*x.shape))
    plt.show()
    '''

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
        self.raw_paths = pd.read_csv(self.csv_raw).to_numpy().squeeze()
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

    def __len__(self):
        return len(self.raw_paths)

    def __getitem__(self, index):

        raw_target, fs = read_wav(self.raw_paths[index])  # Ground truth

        # Transform the raw ground truth
        if self.target_raw_tf is not None:
            self.target_raw_tf(raw_target, **self.target_raw_tf_kwargs)

        # Add noise to the raw ground truth to create the raw input
        raw_in = raw_target
        if self.in_raw_tf is not None:
            raw_in = self.in_raw_tf(raw_in, **self.in_raw_tf_kwargs)

        # Exctract features
        if self.features_tf is not None:
            # TODO fix size
            x, y = (self.features_tf(a, **self.features_tf_kwargs)
                    for a in (raw_in, raw_target))
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
        with open(self.csv_noise, 'r') as f:
            self.noise_paths = f.read().split('\n')
        self.nb_samples_cumsum = self.__init_nb_samples_cumsum()

        # for random noise generation
        import time
        torch.random.manual_seed(int(1e9*time.time()))

        self.resampler = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=16000)

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

        noise, fs = read_wav(self.noise_paths[index])
        #noise = noise.squeeze()
        # resample to the dataset wanted fs `self.fs`
        if (self.fs is not None) and (self.fs != fs):
            if self.resampler.orig_freq != fs:
                self.resampler.orig_freq = fs
        noise = self.resampler(noise)

        return noise

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
        cnt = 0
        while True:  # do ... while
            index = torch.randint(low=0, high=self.nb_samples_cumsum[-1],
                                  size=(1,))
            # convert index into sound_index + sample_index
            sound_index = np.searchsorted(self.nb_samples_cumsum, index)
            sample_index = index - self.nb_samples_cumsum[sound_index]

            # check if there are enough samples at the right of the sound
            if (self.nb_samples_cumsum[sound_index] - index) > noise_len:
                break

            cnt += 1
            if cnt > 100:
                print('STUCK')

        # TODO read with open + struct instead of loading the whole file
        noise = self[sound_index][:, sample_index:sample_index+noise_len]
        if fs is not None:
            self.fs = fs_old

        return noise

    def add_noise_snr(self, sig, *, fs=None, snr):
        # `sig` must be C * T * F
        noise = self.gen_noise(noise_len=sig.shape[1], fs=fs)
        return add_noise_snr(sig, noise, snr)


##################################################
# Functions

def cal_adjusted_rms(clean_rms, snr):
    """ Adjusting RMS to SNR"""
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms


def cal_rms(amp):
    """ Computing root mean square of signal"""
    return np.sqrt(np.mean(np.square(amp), axis=-1))


def add_noise_snr(sig, noise, snr):
    """ Adding noise to sig according to certain SNR"""

    sig = sig.numpy()
    noise = noise.numpy()

    # Recentrer les sons
    clean = sig - np.mean(sig)
    noise = noise - np.mean(noise)

    # Calcul rms son clean
    clean_rms = cal_rms(clean)

    start = random.randint(0, len(noise)-len(clean))
    divided_noise = noise[start: start + len(clean)]

    # Calcul rms bruit
    noise_rms = cal_rms(divided_noise)

    # Ajustement rms bruit pour snr voulu
    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
    adjusted_noise = divided_noise * (adjusted_noise_rms / noise_rms)

    # Ajout bruit au signal
    mixed = (clean + adjusted_noise)

    # Equilibrage rms sortie = rms entree
    mixed = mixed * (clean_rms / cal_rms(mixed))

    # Normalisation dans [-1, 1]
    mixed = mixed/(max(np.amax(mixed), np.amin(mixed)))

    return torch.tensor(mixed) # TODO handle dtype


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

    pd.DataFrame(data={"col1": wav_paths_train}).to_csv(train_path,
                                                        header=None, index=False)
    pd.DataFrame(data={"col1": wav_paths_test}).to_csv(test_path,
                                                       header=None, index=False)

    return train_path, test_path


def stft(x, **kwargs):
    """https://librosa.github.io/librosa/generated/librosa.core.stft.html#librosa-core-stft"""
    S = torch.tensor(np.abs(librosa_stft(x[0].numpy(), **kwargs)),
                     dtype=torch.double).unsqueeze_(dim=0)
    return normalize(S, (S.mean(),), (S.std(),))

##################################################
# Main


if __name__ == '__main__':
    if False:
        csv_noise_train = './data/train_noise.csv'
        NS = NoiseDataset(csv_noise_train, fs=16000)

        noise = NS[0]

        noise = noise[0].numpy()

        noise = noise / (max(np.amax(noise), np.amin(noise)))

        # print(noise[0].numpy())
        wave.write('./data/noise/babble_res.wav', 16000, noise)

    main()
