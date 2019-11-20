import os
import torch
from torch.utils.data import Dataset
from torchaudio.compliance.kaldi import spectrogram


def main():
    root_dir = './data/raw'

    wav_paths_train = []
    for root, dirs, files in os.walk(root_dir):
        if ('train' in root) and (files is not None) and any(f.endswith('.wav') for f in files):
            wav_files = [f for f in files if f.endswith('.wav')]
            wav_paths_train += [os.path.join(root, f) for f in wav_files]

    with open('./train.csv', 'w') as f:
        f.write('\n'.join(wav_paths_train))

    wav_paths_test = []
    for root, dirs, files in os.walk(root_dir):
        if ('test' in root) and (files is not None) and any(f.endswith('.wav') for f in files):
            wav_files = [f for f in files if f.endswith('.wav')]
            wav_paths_test += [os.path.join(root, f) for f in wav_files]

    with open('./test.csv', 'w') as f:
        f.write('\n'.join(wav_paths_test))

    train_set = CustomDataset('./test.csv', transform=spectrogram, target_transform=spectrogram)
    x, y = train_set[0]
    print(type(x))
    print(type(y))



class CustomDataset(Dataset):
    """TIMIT dataset."""

    def __init__(self, csv_paths, transform=None, target_transform=None):
        self.transform = transform
        with open(csv_paths, 'r') as f:
            self.wav_paths = f.read().split('\n')

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, index):

        audio = torch.load(self.wav_paths[index])
        target = audio

        if self.transform is not None:
            audio = self.transform(audio)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return audio, target


if __name__ == '__main__':
    main()
