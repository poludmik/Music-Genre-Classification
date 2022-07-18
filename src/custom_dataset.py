import os
from pydub import AudioSegment
import csv


class songsManager:

    @staticmethod
    def add_songs_from_folder_to_dataset(foldername, to_folder):
        # take all songs from a folder
        # convert them to .wav and to a mono channeling
        # cut 30 second slice from the middle and store it to the to_folder

        for filename in os.listdir(foldername):
            f = os.path.join(foldername, filename)
            if os.path.isfile(f):
                print(f)
                song = AudioSegment.from_file(f)

                song_length_sec = song.duration_seconds
                middle_time_minus_15_sec_in_ms = (song_length_sec // 2 - 15) * 1000
                middle_time_plus_15_sec_in_ms = (song_length_sec // 2 + 15) * 1000

                song_slice = song[middle_time_minus_15_sec_in_ms:middle_time_plus_15_sec_in_ms]

                mono_audios = song_slice.split_to_mono()

                mono_audios[0].export(to_folder + "/" + filename[:-4] + ".wav", format="wav")





if __name__ == "__main__":

    labels = {0: "classical", 1: "pop", 2: "rap", 3: "metal"}

    folder_classical = "C:/Users/micha/homeworks/personal/Music/data/mishas_dataset/downloaded_songs/rap"
    end_folder = "C:/Users/micha/homeworks/personal/Music/data/mishas_dataset/downloaded_songs/rap_30s"
    csv_doc = "C:/Users/micha/homeworks/personal/Music/data/mishas_dataset/labels.csv"

    songsManager.add_songs_from_folder_to_dataset(folder_classical, end_folder)







