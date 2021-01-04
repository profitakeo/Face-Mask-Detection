# Face Mask Detection

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Convolutional Neural Network for detecting the face masks on people

class NetConv(nn.Module):
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64*32*32, 3) # 128 might need to be changed

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        # print(x.shape)
        
        # x = x.view(-1, 128) # 128 might need to be changed
        # x = x.view(x.size(0), -1)
        x = x.view(-1, 64*32*32)
        x = F.relu(self.fc1(x))

        x = F.log_softmax(x, dim=1)
 
        return x

network = NetConv()
optimiser = optim.Adam(network.parameters(), lr=0.00005)
criterion = nn.CrossEntropyLoss()
epochs = 15
batchSize = 16
testSplit = 0.2