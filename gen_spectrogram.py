import numpy as np
import scipy.io.wavfile as wave
import python_speech_features as psf
from pydub import AudioSegment
import matplotlib.pyplot as plt
from PIL import Image

#your sound file
filepath = '/home/abert/Documents/PHELMA/Projet/speech_enhancement/data/raw/TIMIT_TRAIN/DR1/FCJF0/SA1.wav'

def convert(path):

    #open file (supports all ffmpeg supported filetypes)
    audio = AudioSegment.from_file(path, path.split('.')[-1].lower())

    #set to mono
    audio = audio.set_channels(1)

    #set to 44.1 KHz
    audio = audio.set_frame_rate(44100)

    #save as wav
    audio.export(path, format="wav")

def getSpectrogram(sig2, winlen=0.032, winstep=0.016, NFFT=512, rate = 16000):


    #print('taille fen= ',winlen*rate)
    #print('step fen = ',winstep*rate)

    #get frames
    winfunc=lambda x:np.ones((x,))
    frames = psf.sigproc.framesig(sig2, winlen*rate, winstep*rate, winfunc)

    #print(frames.shape)
    #Magnitude Spectrogram
    magspec = np.rot90(psf.sigproc.magspec(frames, NFFT))

    #noise reduction (mean substract)
    magspec -= magspec.mean(axis=0)

    #normalize values between 0 and 1
    magspec -= magspec.min(axis=0)
    magspec /= magspec.max(axis=0)

    #show spec dimensions
    #print(magspec.shape)

    im_spec = Image.fromarray(255*magspec)
    #Converting in B/W mode
    im_spec = im_spec.convert("L")

    return im_spec


if __name__ == '__main__':
    # Get Spectrogram
    (rate,sig) = wave.read(filepath)
    spec = getSpectrogram(sig)
    spec.save("/home/abert/Documents/PHELMA/Projet/speech_enhancement/data/spectrogram/SA1_spec.png")
