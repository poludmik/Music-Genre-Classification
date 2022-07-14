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
from soundtools import SoundTools


df = "C:/Users/micha/homeworks/personal/Music/Data"


audio_file = df + '/genres_original/metal/metal.00009.wav'
sig, sr = torchaudio.load(audio_file)
audio = (sig, sr)

# audio = SoundTools.rechannel(audio, 2)
# sig, sr = audio
# print(sig)

# 22050
new_sr = 22050
audio = SoundTools.resample(audio, new_sr)
sig, sr = audio


print(sig[0].numpy())
print(sr)
print(sig.size())

t = np.linspace(0, 1, new_sr, endpoint = False)
plt.plot(t, sig[0].numpy()[0:new_sr])
plt.show()







#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

labels = pd.read_csv(df + "/features_3_sec.csv")

# print(labels)
# print(labels.iloc[3, 0], labels.iloc[3, 59])

labels = {0: "blues", 1: "classical", 2: "country", 3: "disco", 4: "hiphop",
          5: "jazz", 6: "metal", 7: "pop", 8: "reggae", 9: "rock"}


subfolders = [f.path for f in os.scandir(df + '/genres_original') if f.is_dir()]

idx_overall = 0
chunk_length_ms = 3000

'''''''''
for directory in subfolders:
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            myaudio = AudioSegment.from_file(f, "wav")
            chunks = make_chunks(myaudio, chunk_length_ms)
            chunks = chunks[0:10]
            for i, chunk in enumerate(chunks):
                # labels.iloc[idx_overall, 0][:-3]
                chunk_name = df + "/three_second_samples/" + filename[:-4] + "." + str(i) + ".wav"
                chunk.export(chunk_name, format="wav")

                chunk_name_same = df + "/same_name_3seconds/" + "song" + str(idx_overall) + ".wav"
                chunk.export(chunk_name_same, format="wav")
                idx_overall += 1
'''''''''





