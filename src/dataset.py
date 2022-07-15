from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio
from torchaudio import transforms


class songsDS(Dataset):
    def __init__(self, data_path):
        self.data_path = str(data_path)
        self.duration = 3000
        self.sr = 22050
        self.length = 10000
        self.channel = 1
        self.shift_pct = 0.4

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return self.length

    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):

        audio_file = self.data_path + "song" + str(idx) + ".wav"

        # Get the Class ID
        class_id = idx // 1000

        sig, sr = torchaudio.load(audio_file)
        aud = (sig, sr)

        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)

        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return aug_sgram, class_id
