import numpy as np

from soundtools import SoundTools
from dataset import songsDS
from model import NeuralNetModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



class TrainingAssistant:

    labels = {0: "blues", 1: "classical", 2: "country", 3: "disco", 4: "hiphop",
              5: "jazz", 6: "metal", 7: "pop", 8: "reggae", 9: "rock"}

    def __init__(self):
        pass

    @staticmethod
    def plot_train_and_val_losses(tr_losses, vl_losses, epoch_number):
        epoch_list = list(range(0, epoch_number + 1))
        plt.plot(epoch_list, tr_losses, '-b', label='train loss')
        plt.plot(epoch_list, vl_losses, '-r', label='val loss')
        plt.legend(loc="upper right")
        plt.xlabel("Epoch number")
        plt.ylabel("Average CrossEntropyLoss")
        plt.suptitle("Train and Val loss progression")
        plt.show()

    @staticmethod
    def train(weights_path=None, batch_size=16, lr=0.001, epochs=50, save_dir=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(device)

        if not isinstance(save_dir, str):
            print("Directionary for saving weights is None.")

        plt.style.use('seaborn-whitegrid')

        lr = lr
        batch_size = batch_size
        epochs = epochs

        trainDS = songsDS(train=True)
        valDS = songsDS(validate=True)

        # Create training and validation data loaders
        trainDL = torch.utils.data.DataLoader(trainDS, batch_size=batch_size, shuffle=True)
        valDL = torch.utils.data.DataLoader(valDS, batch_size=batch_size, shuffle=True)

        model = NeuralNetModel()
        model.to(device)
        if isinstance(weights_path, str):
            model.load_state_dict(torch.load(weights_path))

        loss_criterium = nn.CrossEntropyLoss()  # CrossEntropyLoss is for predictions of probabilities, in range [0, 1]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses = []
        val_losses = []

        min_loss = 666

        for epoch in range(epochs):

            sum_of_train_losses = sum_of_val_losses = 0

            for data in trainDL:
                # data[0] is a batch of batch_size spectograms-tensors of shape [1, 90, 260]
                # data[1] is a batch of corresponding labels, e.g. tensor([5, 7, 8, 1, 3, 8, 0, 5, 0, 7, 9, 6, 5, 8, 4, 1])
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                # standartization
                # images = (images - torch.mean(images)) / torch.std(images)

                predictions = model(images)

                loss = loss_criterium(predictions, labels)

                sum_of_train_losses += loss.item()

                # Do the backward pass and update the gradients
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            for data in valDL:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                predictions = model(images)
                loss = loss_criterium(predictions, labels)
                sum_of_val_losses += loss.item()
                optimizer.zero_grad()

            mean_train_loss = sum_of_train_losses / (trainDS.length // batch_size)
            train_losses.append(mean_train_loss)

            mean_val_loss = sum_of_val_losses / (valDS.length // batch_size)
            val_losses.append(mean_val_loss)

            print("Ep:%d, Mean train loss:%.4f, Mean val loss:%.4f." % (epoch, mean_train_loss, mean_val_loss))
            TrainingAssistant.plot_train_and_val_losses(train_losses, val_losses, epoch_number=epoch)

            if min_loss > mean_val_loss and isinstance(save_dir, str):
                torch.save(model.state_dict(), save_dir + "/weights_ep" + str(epoch) + "_loss" + str(mean_val_loss) + ".pth")
                min_loss = mean_val_loss

    @staticmethod
    def test_on_custom_audio(weights_dir):

        if weights_dir is None:
            print("No weights path given.")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        testDS = songsDS(test=True)
        testDL = torch.utils.data.DataLoader(testDS, batch_size=1, shuffle=False)
        model = NeuralNetModel()
        model.to(device)
        model.load_state_dict(torch.load(weights_dir))

        for data in testDL:

            images, _ = data
            images = images.to(device)

            # standartization
            # images = (images - torch.mean(images)) / torch.std(images)

            predictions = model(images)

            print(predictions.tolist())
            n = np.array(predictions.tolist())
            index = np.argmax(n)
            print(index)
            print(f'Argmax index is: {index}, which is {TrainingAssistant.labels[index]}.')



if __name__ == "__main__":

    weights = None
    # weights = "C:/Users/micha/homeworks/personal/Music/data/weights/weights_ep46_loss1.4809265176924244.pth"
    save_directionary = "C:/Users/micha/homeworks/personal/Music/data/weights"


    # TrainingAssistant.test_on_custom_audio(weights)

    # """""""""
    TrainingAssistant.train(weights_path=weights,
                            batch_size=16,
                            lr=0.001,
                            epochs=50,
                            save_dir=save_directionary)
    # """""""""
















