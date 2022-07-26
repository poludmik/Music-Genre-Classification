import random
from pydub import AudioSegment
from pydub.utils import make_chunks
import csv
import os


class songsManager:
    """
    Contains methods that were used to create a custom dataset.
    """

    @staticmethod
    def add_songs_from_folder_to_dataset(foldername, to_folder):
        """
        Take all songs from a folder.
        Convert them to .wav and to a mono channeling.
        Cut 30 second slice from the middle and store it to the to_folder.

        :param str foldername: Absolute path to a folder.
        :param str to_folder: Absolute path to a store folder.
        """
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
    def cut_30s_to_3s_and_store_with_labels(from_folder, lbl, to_folder, csv_file):
        """
        Cut 30-second tracks to 3 seconds and store them with corresponding labels.

        :param str from_folder: Absolute path to a folder.
        :param int lbl: Current label.
        :param str to_folder: Absolute path to a store folder.
        :param str csv_file: Absolute path to a csv file with labels.
        """
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
                    track_name = str(lbl) + "_" + filename[:-4] + "_" + str(i) + ".wav"
                    chunk_name = to_folder + "/" + track_name
                    chunk.export(chunk_name, format="wav")
                    csv_rows.append([track_name, str(lbl)])


        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)

    @staticmethod
    def separate_to_train_and_validation_data(all_folder, train_folder, val_folder, csv_file, csv_train, csv_val):
        """
        Randomly select tracks for validation and separate them from training tracks.

        :param str all_folder: Absolute path to a common folder.
        :param str train_folder: Absolute path to a training folder.
        :param str val_folder: Absolute path to a validation folder.
        :param str csv_file: Absolute path to a common csv file.
        :param str csv_train: Absolute path to a csv training file.
        :param str csv_val: Absolute path to a csv validation file.
        """
        nums = random.sample(range(5999), 250)

        the_file = open(csv_file, 'r')
        reader = csv.reader(the_file)

        data_val = [["songname", "label"]]
        data_train = [["songname", "label"]]

        for i, row in enumerate(reader):

            if i == 0:
                continue

            name = "track_" + str(i) + ".wav"

            if i in nums:
                print(i, row[0])
                os.rename(os.path.join(all_folder, row[0]), os.path.join(val_folder, name))
                data_val.append([name, row[1]])
            else:
                os.rename(os.path.join(all_folder, row[0]), os.path.join(train_folder, name))
                data_train.append([name, row[1]])

        with open(csv_train, 'w', newline='') as f_train, open(csv_val, 'w', newline='') as f_val:
            writer = csv.writer(f_train)
            writer.writerows(data_train)
            writer = csv.writer(f_val)
            writer.writerows(data_val)


if __name__ == "__main__":

    labels = {0: "classical", 1: "pop", 2: "rap", 3: "lofi", 4:"metal"}

    # Trim to 30 seconds
    """
    # folder_classical = "C:/Users/micha/homeworks/personal/Music/data/mishas_dataset/downloaded_songs/lofi"
    # end_folder = "C:/Users/micha/homeworks/personal/Music/data/mishas_dataset/downloaded_songs/lofi_30s"
    # songsManager.add_songs_from_folder_to_dataset(folder_classical, end_folder)
    """

    # Split to 3s tracks and store them with labels
    """
    label = 3
    csv_doc = "C:/Users/micha/homeworks/personal/Music/data/mishas_dataset/labels_" + labels[label] + ".csv"
    start_folder = "C:/Users/micha/homeworks/personal/Music/data/mishas_dataset/downloaded_songs/" + labels[label] +"_30s"
    end_folder = "C:/Users/micha/homeworks/personal/Music/data/mishas_dataset/downloaded_songs/" + labels[label] +"_3s"

    songsManager.cut_30s_to_3s_and_store_with_labels(start_folder, label, end_folder, csv_doc)
    """

    # Separate to val and train data
    """
    csv_doc = "C:/Users/micha/homeworks/personal/Music/data/mishas_custom_dataset/custom_all_3s_labels.csv"
    folder_of_all = "C:/Users/micha/homeworks/personal/Music/data/mishas_custom_dataset/custom_all_3s_tracks_train"
    folder_of_training = "C:/Users/micha/homeworks/personal/Music/data/mishas_custom_dataset/custom_training_folder"
    folder_of_validation = "C:/Users/micha/homeworks/personal/Music/data/mishas_custom_dataset/custom_validation_folder"
    csv_doc_train = "C:/Users/micha/homeworks/personal/Music/data/mishas_custom_dataset/training_labels.csv"
    csv_doc_val = "C:/Users/micha/homeworks/personal/Music/data/mishas_custom_dataset/validation_labels.csv"

    songsManager.separate_to_train_and_validation_data(folder_of_all, folder_of_training, folder_of_validation, csv_doc, csv_doc_train, csv_doc_val)
    """
