# PREPROCESSING

# Importing libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torchvision
import cv2
import model
from model import NetConv
import argparse
import time


import os
from bs4 import BeautifulSoup

# Loading data

class faceMaskDataset(Dataset):
    def __init__(self, img_folder, annot_folder, transform=None):
        # Extracting image name and class from xml file
        desc = []
        for dirname, _, filenames in os.walk(annot_folder):
            for filename in filenames:
                desc.append(os.path.join(dirname, filename))

        img_name,label = [],[]

        for d in desc:
            content = []
            n = []

            with open(d, "r") as file:
                content = file.readlines()
            content = "".join(content)
            soup = BeautifulSoup(content,"html.parser")
            file_name = soup.filename.string
            name_tags = soup.find_all("name")
            
            for t in name_tags:
                n.append(t.get_text())
                
            # selecting tag with maximum occurence in an image (If it has multiple tags)
            name = max(set(n), key = n.count)
        
            img_name.append(file_name)
            label.append(name)

        labels = pd.get_dummies(label)
        print(labels.head())

        # Our target classes
        classes = list(labels.columns)
        print(classes)

        data, target = [],[]
        img_h, img_w = 256, 256

        # Loading images and converting them to pixel array
        for i in range(len(img_name)):
            name = os.path.join("./images/1", img_name[i])
            image = cv2.imread(name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (img_w, img_h), cv2.INTER_AREA)
            
            data.append(image)
            target.append(tuple(labels.iloc[i,:]))

        print(type(data))
        # data = np.array(data)
        data = np.array(data) / 255 # Normalise pixel data to between 0 and 1
        target = np.array(target)
        
        data = np.swapaxes(data, 1, 3)
        data = np.swapaxes(data, 2, 3)

        self.data = data
        print("target len")
        print(len(target))

        print("data shape target shape")
        print(data.shape, target.shape)
        
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        image = self.data[index,:]
        label = torch.tensor(self.target[index])
        return (image, label)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda',action='store_true',default=False,help='disables CUDA')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    img_folder = "./images/1"
    annot_folder = "./annotations"

    optimiser = model.optimiser
    criterion = model.criterion
    testSplit = model.testSplit
    batchSize = model.batchSize

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    dataset = faceMaskDataset(img_folder, annot_folder)
    print((round(853*testSplit) + round(853*(1-testSplit))))
    train_set, val_set = torch.utils.data.random_split(dataset, [round(853*testSplit), round(853*(1-testSplit))])
    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=True, batch_size=batchSize, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(dataset=val_set, shuffle=True, batch_size=batchSize, num_workers=4)

    train(NetConv().to(device), optimiser, nn.CrossEntropyLoss(), train_loader, validation_loader, model.epochs, "cuda")

    return

########################################
# https://discuss.pytorch.org/t/valueerror-expected-input-batch-size-324-to-match-target-batch-size-4/24498 maybe use this implementation for training and testing
########################################

########################################


def train(model, optimizer, loss_fn, train_dl, val_dl, epochs=20, device='cuda'):
    '''
    Parameters
    ----------
    model : nn.Module
        Neural network model
    optimizer : torch.optim.Optimizer
        Search space optimizer (e.g. Adam)
    loss_fn :
        Loss function (e.g. nn.CrossEntropyLoss())
    train_dl : 
        Iterable dataloader for training data.
    val_dl :
        Iterable dataloader for validation data.
    epochs : int
        Number of epochs to run
    device : string
        Specifies 'cuda' or 'cpu'
    '''

    print('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, device))

    history = {} # Collects per-epoch loss and acc like Keras' fit().
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []

    start_time_sec = time.time()

    for epoch in range(epochs):

        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        train_loss         = 0.0
        num_train_correct  = 0
        num_train_examples = 0

        for batch in train_dl:
            optimizer.zero_grad()

            x = batch[0]
            x = x.to(device)

            y = batch[1]
            y = y.to(device)

            yhat = model(x.float())

            loss = loss_fn(yhat, torch.max(y, 1)[1]) # https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216/5

            loss.backward()
            optimizer.step()

            train_loss         += loss.data.item() * x.size(0)
            num_train_correct  += (torch.max(yhat, 1)[1] == torch.max(y, 1)[1]).sum().item()
            print("Num Correct: %2d" % (num_train_correct))
            num_train_examples += x.shape[0]

        train_acc   = num_train_correct / num_train_examples
        train_loss  = train_loss / len(train_dl.dataset)
        print("Epoch: %2d, Loss: %.3f, Acc: %.3f" % (epoch, train_loss, train_acc))

        


    # --- EVALUATE ON VALIDATION SET -------------------------------------
    model.eval()
    val_loss       = 0.0
    num_val_correct  = 0
    num_val_examples = 0

    for batch in val_dl:

        x = batch[0]
        x = x.to(device)

        y = batch[1]
        y = y.to(device)

        yhat = model(x.float())
        loss = loss_fn(yhat, torch.max(y, 1)[1])

        val_loss         += loss.data.item() * x.size(0)
        num_val_correct  += (torch.max(yhat, 1)[1] == torch.max(y, 1)[1]).sum().item()
        num_val_examples += y.shape[0]

    val_acc  = num_val_correct / num_val_examples
    val_loss = val_loss / len(val_dl.dataset)


    print('Epoch %3d/%3d, train loss: %.6f, train acc: %.6f, val loss: %.6f, val acc: %.6f' % \
            (epoch+1, epochs, train_loss, train_acc, val_loss, val_acc))

    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['acc'].append(train_acc)
    history['val_acc'].append(val_acc)

# END OF TRAINING LOOP


    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    return history

    # print('Finished Training')

if __name__ == '__main__':
    main()
