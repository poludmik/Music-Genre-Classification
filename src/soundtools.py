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

    @staticmethod
    def open(sound_file):
        """
        Opens the audio file.

        :param str sound_file: Absolute path to a sound file.
        :return: Tuple (signal, sample_rate), where signal is a torch tensor.
        """
        signal, sample_rate = torchaudio.load(sound_file)
        return signal, sample_rate

    @staticmethod
    def rechannel(sound, new_num_channels):
        """
        Change the number of channels in an audio.

        :param tuple sound: An audio signal.
        :param int new_num_channels: A new number of audio channels.
        :return: New sound tuple (new_signal, sample_rate), where signal is a torch tensor.
        """
        (signal, sample_rate) = sound
        if signal.shape[0] == new_num_channels:
            return sound
        if new_num_channels == 1:
            return signal[0].view(1, signal[0].size(dim=0)), sample_rate
        if new_num_channels == 2:
            return torch.cat((signal, signal)), sample_rate

    @staticmethod
    def resample(sound, new_sample_rate):
        """
        Change number of samples per seconds in sound.

        :param tuple sound: An audio signal.
        :param int new_sample_rate: Sample rate will be changed to this number.
        :return: New sound tuple (signal, new_sample_rate), where signal is a torch tensor.
        """
        signal, sample_rate = sound
        if sample_rate == new_sample_rate:
            return sound

        transform = transforms.Resample(sample_rate, new_sample_rate)
        signal = transform(signal)
        if signal.size(dim=0) == 2:
            return torch.cat((signal, signal)), new_sample_rate
        return signal, new_sample_rate

    @staticmethod
    def sec_to_sn(t_sec, sr):
        """
        Convert time in seconds to number of samples using sample rate.

        :param float t_sec: Time in seconds.
        :param int sr: Sample rate.
        :return: Number of samples during t_sec.
        """
        return math.floor(t_sec * sr)

    @staticmethod
    def sn_to_sec(sn, sr):
        """
        Convert number of samples to time in seconds using sample rate.

        :param int sn: Number of samples.
        :param int sr: Sample rate.
        :return: Time in seconds.
        """
        return math.ceil((sn / sr))

    @staticmethod
    def cut_or_pad(sound, t_sec):
        """
        Pad with zeros or cut the given sound, so that it is t_sec long.

        :param tuple sound: A sound tuple (signal_tensor, sample_rate).
        :param int t_sec: Desired length in seconds.
        :return: Cut or padded sound tuple (new_signal_tensor, sample_rate).
        """
        signal, sample_rate = sound
        samples_number = SoundTools.sec_to_sn(t_sec, sample_rate)
        if signal.size(dim=1) == samples_number:
            return sound
        elif signal.size(dim=1) > samples_number:
            signal = signal[:, :samples_number]
        elif signal.size(dim=1) < samples_number:
            difference = samples_number - signal.size(dim=1)
            channels = signal.size(dim=0)
            if difference % 2 == 0:
                pad_start = torch.zeros((channels, difference // 2))
                pad_end = pad_start
            else:
                pad_start = torch.zeros((channels, difference // 2))
                pad_end = torch.zeros((channels, difference - difference // 2))
            signal = torch.cat((pad_start, signal, pad_end), dim=1)

        return signal, sample_rate

    @staticmethod
    def plot_sound(sound):
        """
        Shows the sound in time domain using matplotlib.

        :param tuple sound: A sound (signal_tensor, sample_rate).
        """
        sig, sr = sound
        time_in_s = SoundTools.sn_to_sec(sig.size(dim=1), sr)
        t = np.linspace(0, time_in_s, sr * time_in_s, endpoint=False)
        plt.plot(t, sig[0].numpy()[0:sr * time_in_s])
        plt.show()

    @staticmethod
    def random_shift(sound, max_seconds):
        """
        Used for data augmentation. Shift signal by a random time.

        :param tuple sound: A sound (signal_tensor, sample_rate).
        :param float max_seconds: Max time shift.
        :return: Shifted sound (new_signal_tensor, sample_rate).
        """
        sig, sr = sound
        seconds_to_shift = random.uniform(-max_seconds, max_seconds)
        sample_num_to_shift = SoundTools.sec_to_sn(seconds_to_shift, sr)
        sig = torch.roll(sig, sample_num_to_shift, 1)
        return sig, sr

    @staticmethod
    def spectrogram(sound, n_mels=90, n_fft=800, hop_len=None):
        """
        Extracts the spectrogram out of a sound.

        :param tuple sound: A sound (signal_tensor, sample_rate).
        :param int n_mels: Number of Mel levels.
        :param int n_fft: Number of FFT per frame.
        :param int hop_len: Jump between frames.
        :return: Mel spectrogram in dB.
        """
        sig, sr = sound
        max_db = 50
        spectrum = transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        return transforms.AmplitudeToDB(top_db=max_db)(spectrum)

    @staticmethod
    def plot_spectogram(spectogram, songname):
        """
        Plots spectrogram with matplotlib.

        :param spectogram: Spectrogram to plot.
        :param str songname: A songname to be written above spectrogram image.
        """
        # will plot one channel spectogram
        spectr = spectogram[0]
        plt.title(songname)
        plt.imshow(spectr)
        plt.show()

    @staticmethod
    def shadow_spectr_segment(spectrogram):
        """
        Used for data augmentation.

        :param spectrogram: A spectrogram to be changed.
        :return: Spectrogram with shaded parts.
        """
        # For data augmentation
        shadow_value = spectrogram.mean() + random.uniform(-1, 1)
        max_time_percent = 0.08
        max_freq_percent = 0.08

        number_of_time_shadows = 3
        number_of_freq_shadows = 2

        for i in range(number_of_time_shadows):
            spectrogram = transforms.TimeMasking(max_time_percent * spectrogram.size(dim=2))(spectrogram, shadow_value)

        for i in range(number_of_freq_shadows):
            spectrogram = transforms.FrequencyMasking(max_freq_percent * spectrogram.size(dim=1))(spectrogram, shadow_value)

        return spectrogram

