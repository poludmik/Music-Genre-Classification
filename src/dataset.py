import os
import random
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio
from torchaudio import transforms
from soundtools import SoundTools


class songsDS(Dataset):
    def __init__(self, train=False, validate=False, test=False):
        self.data_folder = "C:/Users/micha/homeworks/personal/Music/data"
        self.train = train
        self.validate = validate
        self.test = test

        if test:
            print("Testing mode.")
            self.data_path = self.data_folder + "/testing_folder/"
            self.length = 1
        elif validate:
            print("Validation mode.")
            self.data_path = self.data_folder + "/val_indexed/"
            self.df = pd.read_csv(self.data_folder + "/val_labels.csv")
            self.length = 500
        else:
            print("Training mode.")
            self.data_path = self.data_folder + "/train_indexed/"
            self.df = pd.read_csv(self.data_folder + "/train_labels.csv")
            self.length = 9500


        self.duration_sec = 3
        self.sr = 22050
        self.channel = 1
        self.max_shift_sec = 0.3


    def __len__(self):
        return self.length

    # Getting idx'th sample from dataset
    def __getitem__(self, idx):

        if idx >= self.length:
            print("Index is bigger than songs amount, error.")

        if self.test:
            filenames = os.listdir(self.data_path)
            sig, sr = torchaudio.load(self.data_path + "/" + filenames[0])
            class_id = -1
        else:
            audio_file = self.data_path + self.df.loc[idx, 'filename']
            class_id = self.df.loc[idx, 'label']
            sig, sr = torchaudio.load(audio_file)

        sound = (sig, sr)

        sound = SoundTools.resample(sound, self.sr)
        sound = SoundTools.rechannel(sound, self.channel)

        sound = SoundTools.cut_or_pad(sound, self.duration_sec)
        if not (self.test or self.validate):
            sound = SoundTools.random_shift(sound, self.max_shift_sec)

        spectrum = SoundTools.spectrogram(sound)

        if random.randint(0, 10) > 7 and not (self.test or self.validate):
            spectrum = SoundTools.shadow_spectr_segment(spectrum)

        return spectrum, class_id
