import os
import string
from gen_spectrogram import getSpectrogram
from PIL import Image

class DatasetCreator(object):
    """Base class that will create our noised dataset
    """

    def __init__(self):
        pass



    def dataset_looper(self, rootdir):

        for root, dirs, files in os.walk(rootdir, topdown=True):
            for file in files:
                if '.wav' in file:
                    print(root)

                    print(os.path.join(root, file))
                    file_path = os.path.join(root, file)

                    #open wav file
                    (rate,sig) = wave.read(file_path)
                    print(rate)
                    # Create corresponding directory if doesn't exist
                    #new_root = string.replace(, '')

                    spectro = getSpectrogram(sig)

                    spectro.save(file_path[:-4]+'.png')





    def add_noise(self, ):



if __name__ == '__main__':
    DC = DatasetCreator()
    train_data_path = './data/raw/TIMIT_TRAIN'
    DC.dataset_looper(train_data_path)
