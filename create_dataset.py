import os
import string
import scipy.io.wavfile as wave
from gen_spectrogram import getSpectrogram
from create_mixed_audio_file import mix_audio
from PIL import Image




class DatasetCreator(object):
    """Base class that will create our noisy dataset
    """

    def __init__(self):
        pass



    def dataset_looper(self, rootdir):

        # Open noise file
        noise_path = './babble.wav'
        (rate, noise) = wave.read(noise_path)


        # Cycle through dataset
        for root, dirs, files in os.walk(rootdir, topdown=True):
            for file in files:
                if '.wav' in file:
                    print(root)

                    print(os.path.join(root, file))
                    file_path = os.path.join(root, file)

                    #open wav file
                    (rate, sig) = wave.read(file_path)
                    print(rate)
                    # Create corresponding directory if doesn't exist
                    #new_root = string.replace(, '')
                    sig_mixed = mix_audio(sig, noise, 0)

                    spectro = getSpectrogram(sig_mixed)


                    # Changing path to link to noisy folder
                    root2 = root.replace('raw', 'noisy')
                    file_path2 = os.path.join(root2, file)

                    #Create directories in noisy folder
                    if not os.path.exists(root2):
                        os.makedirs(root2)


                    #Saving wav_file
                    print(file_path2, rate, type(sig_mixed))
                    wave.write(file_path2, rate, sig_mixed)

                    # Saving spectrogram
                    spectro.save(file_path2[:-4]+'.png')




if __name__ == '__main__':
    DC = DatasetCreator()
    train_data_path = './data/raw/TIMIT_TRAIN'
    DC.dataset_looper(train_data_path)
