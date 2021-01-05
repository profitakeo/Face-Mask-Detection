# Face Mask Detection

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Convolutional Neural Network for detecting the face masks on people

class NetConv(nn.Module):
    def __init__(self):
        super(NetConv, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels = 8, kernel_size = 3)
        self.conv2 = nn.Conv2d(in_channels= 8, out_channels = 16, kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels= 16, out_channels = 32, kernel_size = 3)

        self.fc1 = nn.Linear(30*30*32, 1000) # 128 might need to be changed
        self.fc2 = nn.Linear(1000, 200)
        self.fc3 = nn.Linear(200, 3)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        # print(x.shape)
        
        # x = x.view(-1, 128) # 128 might need to be changed
        # x = x.view(x.size(0), -1)
        x = x.reshape(-1, 30*30*32)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.softmax(x, dim=1)

        # print(x.shape)
 
        return x

network = NetConv()
optimiser = optim.Adam(network.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()
epochs = 15
batchSize = 16
testSplit = 0.2