from soundtools import SoundTools
from dataset import songsDS
from model import NeuralNetModel
import torch
import torch.nn as nn
import torch.nn.functional as F



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    lr = 0.001
    batch_size = 16

    trainDS = songsDS(train=True)
    valDS = songsDS(train=False)

    # Create training and validation data loaders
    trainDL = torch.utils.data.DataLoader(trainDS, batch_size=batch_size, shuffle=True)
    valDL = torch.utils.data.DataLoader(valDS, batch_size=batch_size, shuffle=True)

    model = NeuralNetModel()
    model.to(device)

    loss_criterium = nn.CrossEntropyLoss() # CrossEntropyLoss is for predictions of probabilities, in range [0, 1]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i, data in enumerate(trainDL):
        # data[0] is a batch of batch_size spectograms-tensors of shape [1, 90, 260]
        # data[1] is a batch of corresponding labels, e.g. tensor([5, 7, 8, 1, 3, 8, 0, 5, 0, 7, 9, 6, 5, 8, 4, 1])
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)

        # print(F.one_hot(labels, num_classes=10))

        loss = loss_criterium(predictions, labels)
        print(loss.item())

        # Do the backward pass and update the gradients
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

















