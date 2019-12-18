# -*- coding: utf-8 -*-

import random
import numpy as np
import scipy.io.wavfile as wave
#import scipy.signal as sigscy
from torchaudio.compliance import kaldi

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms

def cal_rms(amp):

    return np.sqrt(np.mean(np.square(amp), axis=-1))


def mix_audio(clean_wav, noise_wav, snr):

    # Recentrer les sons
    clean = clean_wav - np.mean(clean_wav)
    noise = noise_wav - np.mean(noise_wav)

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

    return mixed




if __name__ == '__main__':

    # Open noise file
    noise_path = './babble.wav'
    (rate1, noise) = wave.read(noise_path)
    print("rate_noise", rate1)


    train_data_path = './data/raw/TIMIT_TRAIN/DR6/FMJU0/SA2.wav'
    (rate2, sig) = wave.read(train_data_path)
    print("rate_sound", rate2)

    #noise2 = kaldi.resample_waveform(noise, rate1, rate2)
    noise2 = sigscy.resample(noise, int(rate2/rate1 * noise.shape[0]) )
    noise2 = noise2/(max(np.amax(noise2), np.amin(noise2)))
    wave.write('./test_add_noise/babble.wav', rate2, noise2)
    noise_path = './test_add_noise/babble.wav'
    (rate3, noise2) = wave.read(noise_path)
    print("rate_noise3", rate3)



    snr = 0

    sig_mixed = mix_audio(sig, noise2, snr)

    wave.write('./test_add_noise/mixed_sound.wav', rate2, sig_mixed)
