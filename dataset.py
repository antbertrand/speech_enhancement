import os
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.compliance.kaldi import spectrogram
import numpy as np
import matplotlib.pyplot as plt

from utils import sec_to_hms


##################################################
# Main

def main():
    root_dir = './data/raw'
    csv_train = './train.csv'
    csv_test = './test.csv'

    train_set = CustomDataset(root_dir, csv_train)
    test_set = CustomDataset(root_dir, csv_test)
    
    _, ax = plt.subplots()
    train_set.histogram_wav_length(ax, label='Train')
    test_set.histogram_wav_length(ax, label='Test')
    
    ax.legend()
    plt.show()


##################################################
# Classes

class CustomDataset(Dataset):
    """TIMIT dataset."""

    def __init__(self, root_dir, csv_path, transform=None, target_transform=None):

        self.root_dir = root_dir
        self.csv_path = csv_path
        self.transform = transform
        self.target_transform = target_transform

        with open(csv_path, 'r') as f:
            self.wav_paths = f.read().split('\n')

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, index):

        audio = torchaudio.load(self.wav_paths[index])
        target = audio

        if self.transform is not None:
            audio = self.transform(audio)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # TODO normaliser

        return audio, target
    
    def histogram_wav_length(self, ax=None, label=None):
        
        print('Computing histogram of lengths.')
        rates = np.zeros(len(self))
        n_samples = np.zeros(len(self))
        for i, wav_path in enumerate(self.wav_paths):
            print('#{:03d}/{:d}'.format(i+1, len(self)), end='\r')
            si, _ = torchaudio.info(wav_path)
            rates[i], n_samples[i] = si.rate, si.length
        
        rate = rates[0]
        if not all(r==rate for r in rates):
            print('WARNING : not all rates are the same')
            return
        
        if ax is None:
            _, ax = plt.subplots()
        n, _, p = ax.hist(n_samples, bins=50)
        ax.set_title('Histogramme des durées')
        xticks = ax.get_xticks()
        ax.set_xticklabels(["{:.0f}\n{:.2f}".format(t, t/rate) for t in xticks])
        ax.set_xlabel("Nombre d'échantillons [] | Durée [s]")
        
        if label is not None:
            p[0].set_label('{} ({:d} samples | {:d}h{:02d}min{:02.0f}s)'.format(label, int(sum(n_samples)), *sec_to_hms(sum(n_samples)/rate)))
        

##################################################
# Functions


def create_csv(root_dir, train_path=None, test_path=None):

    wav_paths_train = []
    wav_paths_test = []

    for root, _, files in os.walk(root_dir):

        if ('train' in root) and (files is not None) and any(f.endswith('.wav') for f in files):
            wav_files = [f for f in files if f.endswith('.wav')]
            wav_paths_train += [os.path.join(root, f) for f in wav_files]

        if ('test' in root) and (files is not None) and any(f.endswith('.wav') for f in files):
            wav_files = [f for f in files if f.endswith('.wav')]
            wav_paths_test += [os.path.join(root, f) for f in wav_files]

    if train_path is None:
        train_path = './train.csv'
    if test_path is None:
        test_path = './test.csv'

    with open(train_path, 'w') as f:
        f.write('\n'.join(wav_paths_train))
    with open(test_path, 'w') as f:
        f.write('\n'.join(wav_paths_test))

    return train_path, test_path


##################################################
# Main


if __name__ == '__main__':
    main()
