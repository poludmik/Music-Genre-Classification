import math, random
import os
import torch
import pandas as pd
import numpy as np
import torchaudio
from torchaudio import transforms
import IPython.display as ipd
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.utils import make_chunks



class SoundTools:

    def __init__(self):
        self.kek = 0

    @staticmethod
    def open(self):
        print("kek")


'''''''''''
    signal, sample_rate = torchaudio.load(sound_file)
        return signal, sample_rate

    @staticmethod
    def rechannel(sound, new_num_channels):
        (signal, sample_rate) = sound
        if signal.shape[0] == new_num_channels:
            return sound

        if new_num_channels == 1:
            return signal[0], sample_rate

        if new_num_channels == 2:
            return torch.cat(signal, signal), sample_rate
            '''''''''

