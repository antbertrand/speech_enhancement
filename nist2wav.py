import os
from sphfile import SPHFile


def main():
    
    root_dir = "./data/raw"

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".WAV"):
                wav_file = os.path.join(root, file)
                nist2wav(wav_file)


def nist2wav(wav_file):

    sph = SPHFile(wav_file)
    txt_file = ""
    txt_file = wav_file[:-3] + "TXT"

    f = open(txt_file,'r')
    for line in f:
        words = line.split(" ")
        start_time = (int(words[0])/16000)
        end_time = (int(words[1])/16000)
    print("writing file ", wav_file)
    sph.write_wav(wav_file.replace(".WAV",".wav"),start_time,end_time)


if __name__ == '__main__':
    main()