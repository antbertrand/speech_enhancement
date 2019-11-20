# -*- coding: utf-8 -*-

import random
import numpy as np



def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 10
    noise_rms = clean_rms / (10**a)
    return noise_rms

def cal_amp(wf):
    # The dtype depends on the value of pulse-code modulation. The int16 is set for 16-bit PCM.
    amptitude = (np.frombuffer(wf, dtype="int16")).astype(np.float64)
    return amptitude

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))


def mix_audio(clean_wav, noise_wav, snr):

    print('TAILLE_SIG',clean_wav.shape[0])
    print('MAX CLEAN',np.amax(clean_wav))
    print('MAX NOISE',np.amax(noise_wav))
    clean_amp = cal_amp(clean_wav)
    noise_amp = cal_amp(noise_wav)

    clean_rms = cal_rms(clean_amp)

    start = random.randint(0, len(noise_amp)-len(clean_amp))
    divided_noise_amp = noise_amp[start: start + len(clean_amp)]
    noise_rms = cal_rms(divided_noise_amp)

    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)

    adjusted_noise_amp = divided_noise_amp * (adjusted_noise_rms / noise_rms)
    mixed_amp = (clean_amp + adjusted_noise_amp)

    #Avoid clipping noise
    '''
    max_int16 = np.iinfo(np.int16).max
    if  mixed_amp.max(axis=0) > max_int16:
        reduction_rate = max_int16 / mixed_amp.max(axis=0)
        mixed_amp = mixed_amp * (reduction_rate)
    '''
    print('MAX MIXED',np.amax(mixed_amp))
    return mixed_amp
