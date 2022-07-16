import torch
import torch.nn as nn
import torch.nn.functional as F

def Conv_Block_and_MaxPool(in_channels, out_channels, kernel=(3, 3), padding=1):
    layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding),
                           nn.BatchNorm2d(out_channels),
                           nn.LeakyReLU(0.1),
                           nn.MaxPool2d(2, stride=2, padding=0)
                           )
    return layers


class NeuralNetModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_block1 = Conv_Block_and_MaxPool(1, 16, (3, 3), padding=1)
        self.conv_block2 = Conv_Block_and_MaxPool(16, 32, (3, 3), padding=1)
        self.conv_block3 = Conv_Block_and_MaxPool(32, 64, (3, 3), padding=1)
        self.conv_block4 = Conv_Block_and_MaxPool(64, 128, (3, 3), padding=1)
        self.conv_block5 = Conv_Block_and_MaxPool(128, 256, (3, 3), padding=1)
        # self.conv_block6 = Conv_Block_and_MaxPool(256, 512, (3, 3), padding=1)

        self.linear1 = nn.Linear(256, 10, bias=True)
        self.linear2 = nn.Linear(4096, 10, bias=True)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)


    def forward(self, x):
        # print(x.shape)
        x = self.conv_block1(x)
        # print(x.shape)
        x = self.conv_block2(x)
        # print(x.shape)
        x = self.conv_block3(x)
        # print(x.shape)
        x = self.conv_block4(x)
        # print(x.shape)
        x = self.conv_block5(x)
        # print(x.shape)
        # x = self.conv_block6(x)
        # print(x.shape)
        # print("flattening")
        # x = torch.flatten(x, 1)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.linear1(x)
        # print(x.shape)
        # x = self.linear2(x)
        # print(x.shape)
        x = F.softmax(x, dim=1)
        # print(x.shape)
        return x


    def weight_init(self):
        for lay in self.modules():
            if type(lay) in [torch.nn.Conv2d, torch.nn.Linear]:
                torch.nn.init.xavier_uniform_(lay.weight)

