import os
import random
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio
from torchaudio import transforms
from soundtools import SoundTools


class songsDS(Dataset):
    def __init__(self, train=True):
        self.data_folder = "C:/Users/micha/homeworks/personal/Music/Data"
        self.train = train

        if self.train:
            self.data_path = self.data_folder + "/train_indexed/"
            self.df = pd.read_csv(self.data_folder + "/train_labels.csv")
            self.length = 9500
        else:
            self.data_path = self.data_folder + "/val_indexed/"
            self.df = pd.read_csv(self.data_folder + "/val_labels.csv")
            self.length = 500

        self.duration_sec = 3
        self.sr = 22050
        self.channel = 1
        self.max_shift_sec = 0.2

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return self.length

    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):

        if idx >= self.length:
            print("Index is bigger than songs amount, error.")

        audio_file = self.data_path + self.df.loc[idx, 'filename']
        class_id = self.df.loc[idx, 'label']
        print(self.df.loc[idx, 'filename'], class_id)

        sig, sr = torchaudio.load(audio_file)
        sound = (sig, sr)

        sound = SoundTools.resample(sound, self.sr)
        sound = SoundTools.rechannel(sound, self.channel)

        sound = SoundTools.cut_or_pad(sound, self.duration_sec)
        sound = SoundTools.random_shift(sound, self.max_shift_sec)
        spectrum = SoundTools.spectrogram(sound)
        if random.randint(0, 10) > 7:
            spectrum = SoundTools.shadow_spectr_segment(spectrum)

        return spectrum, class_id
