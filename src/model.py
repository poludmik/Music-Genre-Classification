import torch
import torch.nn as nn
import torch.nn.functional as F

def Conv_Block_and_MaxPool(in_channels, out_channels, kernel=(3, 3), padding=1):
    """
    A block that connects convolution, batrch norm, relu and maxpool.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param kernel: Size of convolution kernel.
    :param padding: Size of padding.
    :return: A combined convolutional block.
    """
    layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding),
                           nn.BatchNorm2d(out_channels),
                           nn.LeakyReLU(0.1),
                           nn.MaxPool2d(2, stride=2, padding=0)
                           )
    return layers


class NeuralNetModel(nn.Module):
    """
    Subclass implementing the neural network using torch.nn.Module.
    5 convolutional layers,
    1 2D adaptive pooling layer,
    1 fully connected layer.
    """
    def __init__(self):
        super().__init__()
        self.conv_block1 = Conv_Block_and_MaxPool(1, 16, (3, 3), padding=1)
        self.conv_block2 = Conv_Block_and_MaxPool(16, 32, (3, 3), padding=1)
        self.conv_block3 = Conv_Block_and_MaxPool(32, 64, (3, 3), padding=1)
        self.conv_block4 = Conv_Block_and_MaxPool(64, 128, (3, 3), padding=1)
        self.conv_block5 = Conv_Block_and_MaxPool(128, 256, (3, 3), padding=1)

        self.linear1 = nn.Linear(256, 5, bias=True)
        self.linear2 = nn.Linear(4096, 10, bias=True)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)


    def forward(self, x):
        """
        Forward pass on a tensor x.

        :param x: A tensor.
        :return: A tensor.
        """
        # print(x.shape)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = F.softmax(x, dim=1)
        return x


    def weight_init(self):
        """
        Initialize weights on a created model.
        """
        for lay in self.modules():
            if type(lay) in [torch.nn.Conv2d, torch.nn.Linear]:
                torch.nn.init.xavier_uniform_(lay.weight)
