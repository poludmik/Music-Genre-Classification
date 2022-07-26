import random
import pandas as pd
from torch.utils.data import Dataset
import torchaudio
from soundtools import SoundTools


class songsDS(Dataset):
    """
    Subclass of Dataset class implementing a tool for DataLoader to get the data items one at a time.
    """
    def __init__(self, train=False, validate=False, test=False):
        self.data_folder = "C:/Users/micha/homeworks/personal/Music/data/mishas_custom_dataset"
        self.train = train
        self.validate = validate
        self.test = test

        if test:
            print("Testing mode.")
            self.data_path = self.data_folder + "/testing_folder/"
            self.df = pd.read_csv(self.data_folder + "/testing_labels.csv")
            self.length = 30
        elif validate:
            print("Validation mode.")
            self.data_path = self.data_folder + "/custom_validation_folder/"
            self.df = pd.read_csv(self.data_folder + "/validation_labels.csv")
            self.length = 250
        else:
            print("Training mode.")
            self.data_path = self.data_folder + "/custom_training_folder/"
            self.df = pd.read_csv(self.data_folder + "/training_labels.csv")
            self.length = 5750

        self.duration_sec = 3
        self.sr = 44100
        self.channel = 1
        self.max_shift_sec = 1


    def __len__(self):
        """ Return number of items in a dataset.

        :return: Number of items in a dataset.
        """
        return self.length

    def __getitem__(self, idx):
        """
        Getting idx'th sample from dataset.

        :param idx: Index of an item from dataset
        :return: Spectrogram to be used in learning.
        """

        if idx >= self.length:
            print("Index is bigger than songs amount, error.")

        audio_file = self.data_path + self.df.loc[idx, 'songname']
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

        if self.test:
            print(self.df.loc[idx, 'songname'])
        #     SoundTools.plot_spectogram(spectrum, songname)

        return spectrum, class_id
