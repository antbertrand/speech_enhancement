import torch
from torch.utils.data import Dataset


def main():
    root_dir = './data/raw/timit_train'

    for root, dirs, files in os.walk(root_dir):
        pass


class TimitDataset(Dataset):
    """TIMIT dataset."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        pass

    def __getitem__(self, idx):
        sample = None

        return sample


if __name__ == '__main__':
    main()