import os
from pydub import AudioSegment
from pydub.utils import make_chunks
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
                print(filename)
                song = AudioSegment.from_file(f)

                song_length_sec = song.duration_seconds
                middle_time_minus_15_sec_in_ms = (song_length_sec // 2 - 15) * 1000
                middle_time_plus_15_sec_in_ms = (song_length_sec // 2 + 15) * 1000

                song_slice = song[middle_time_minus_15_sec_in_ms:middle_time_plus_15_sec_in_ms]

                mono_audios = song_slice.split_to_mono()

                mono_audios[0].export(to_folder + "/" + filename[:-4] + ".wav", format="wav")


    @staticmethod
    def cut_30s_to_3s_and_store_with_labels(from_folder, label, to_folder, csv_file):

        chunk_length_ms = 3000

        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            csv_rows = list(reader)

        for filename in os.listdir(from_folder):
            f = os.path.join(from_folder, filename)
            if os.path.isfile(f):
                myaudio = AudioSegment.from_file(f, "wav")
                chunks = make_chunks(myaudio, chunk_length_ms)
                chunks = chunks[0:10]
                if len(chunks) < 10:
                    print("Chunks:", len(chunks), filename)

                for i, chunk in enumerate(chunks):
                    track_name = str(label) + "_" + filename[:-4] + "_" + str(i) + ".wav"
                    chunk_name = to_folder + "/" + track_name
                    chunk.export(chunk_name, format="wav")
                    csv_rows.append([track_name, str(label)])


        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)


if __name__ == "__main__":

    labels = {0: "classical", 1: "pop", 2: "rap", 3: "lofi", 4:"metal"}

    # Trim to 30 seconds
    """""""""
    # folder_classical = "C:/Users/micha/homeworks/personal/Music/data/mishas_dataset/downloaded_songs/lofi"
    # end_folder = "C:/Users/micha/homeworks/personal/Music/data/mishas_dataset/downloaded_songs/lofi_30s"
    # songsManager.add_songs_from_folder_to_dataset(folder_classical, end_folder)
    """""""""

    # Split to 3s tracks and store them with labels
    label = 3
    csv_doc = "C:/Users/micha/homeworks/personal/Music/data/mishas_dataset/labels_" + labels[label] + ".csv"
    start_folder = "C:/Users/micha/homeworks/personal/Music/data/mishas_dataset/downloaded_songs/" + labels[label] +"_30s"
    end_folder = "C:/Users/micha/homeworks/personal/Music/data/mishas_dataset/downloaded_songs/" + labels[label] +"_3s"

    songsManager.cut_30s_to_3s_and_store_with_labels(start_folder, label, end_folder, csv_doc)

    # header = ["songname", "label"]
    # data = [["hammer_smashed_face", "500"], ["yellow_submarine", "700"]]
    # with open(csv_doc, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #
    #     # write the header
    #     writer.writerow(header)
    #
    #     # write multiple rows
    #     writer.writerows(data)









