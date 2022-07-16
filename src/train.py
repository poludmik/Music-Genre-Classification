from soundtools import SoundTools
from dataset import songsDS
from model import NeuralNetModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def plot_train_and_val_losses(tr_losses, vl_losses, epoch_number):
    epoch_list = list(range(0, epoch_number + 1))
    plt.plot(epoch_list, tr_losses, '-b', label='train loss')
    plt.plot(epoch_list, vl_losses, '-r', label='val loss')
    plt.legend(loc="upper right")
    plt.xlabel("Epoch number")
    plt.ylabel("Average CrossEntropyLoss")
    plt.suptitle("Train and Val loss progression")
    plt.show()



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    plt.style.use('seaborn-whitegrid')

    lr = 0.001
    batch_size = 16
    epochs = 50

    trainDS = songsDS(train=True)
    valDS = songsDS(train=False)

    # Create training and validation data loaders
    trainDL = torch.utils.data.DataLoader(trainDS, batch_size=batch_size, shuffle=True)
    valDL = torch.utils.data.DataLoader(valDS, batch_size=batch_size, shuffle=True)

    model = NeuralNetModel()
    model.to(device)

    loss_criterium = nn.CrossEntropyLoss() # CrossEntropyLoss is for predictions of probabilities, in range [0, 1]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        sum_of_train_losses = sum_of_val_losses = 0

        for data in trainDL:
            # data[0] is a batch of batch_size spectograms-tensors of shape [1, 90, 260]
            # data[1] is a batch of corresponding labels, e.g. tensor([5, 7, 8, 1, 3, 8, 0, 5, 0, 7, 9, 6, 5, 8, 4, 1])
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

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
        plot_train_and_val_losses(train_losses, val_losses, epoch_number=epoch)
















