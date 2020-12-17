# PREPROCESSING

# Importing libraries
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import model
import argparse
import time


import tensorflow as tf
from keras.preprocessing.image import img_to_array,load_img
#from tensorflow.keras import layers
#from tensorflow.keras.models import Sequential

import os
from bs4 import BeautifulSoup

# Loading data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda',action='store_true',default=False,help='disables CUDA')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    img_folder = "C:/Users/Harry/Desktop/Projects/Face-Mask-Detection/images"
    annot_folder = "C:/Users/Harry/Desktop/Projects/Face-Mask-Detection/annotations"

    # img_folder = "C:/Users/profi/Downloads/Face_Mask_Dataset_(from Kaggle)/images"
    # annot_folder = "C:/Users/profi/Downloads/Face_Mask_Dataset_(from Kaggle)/annotations"

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

    # print(img_name)

    # One Hot Encoding label data
    labels = pd.get_dummies(label)
    print(labels.head())

    # Our target classes
    classes = list(labels.columns)
    print(classes)

    # Loading images and converting them to pixel array
    data, target = [],[]
    img_h, img_w = 256, 256

    for i in range(len(img_name)):
        name = os.path.join("images", img_name[i])
        
        # image = cv2.imread(path, mode='RGB')

        # Try this if above imread doesnt work
        # print(name)
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_w, img_h), cv2.INTER_AREA)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        
        data.append(image)
        # print(data)
        target.append(tuple(labels.iloc[i,:]))

    # Converting list to array
    print(type(data))
    data = np.array(data) / 255
    target = np.array(target)
    print(data)
    # data = np.array(data,dtype = "float32")/255.0
    # target = np.array(target,dtype = "float32")

    # Shape of data and target
    print(data.shape, target.shape)


    # TRAINING
    optimiser = model.optimiser
    criterion = model.criterion
    testSplit = model.testSplit
    batchSize = model.batchSize

    # Splitting into train and test data
    train_img, test_img, y_train, y_test = train_test_split(data,target,test_size=testSplit,random_state=20)

    print("Train shapes : ",(train_img.shape, y_train.shape))
    print("Test shapes : ",(test_img.shape, y_test.shape))

    train(model, optimiser, nn.CrossEntropyLoss(), train_img, y_train, model.epochs, "cuda")

    return

def train(model, optimizer, loss_fn, train_dl, val_dl, epochs=20, device='cuda'):
    '''
    Runs training loop for classification problems. Returns Keras-style
    per-epoch history of loss and accuracy over training and validation data.

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

    Returns
    -------
    Dictionary
        Similar to Keras' fit(), the output dictionary contains per-epoch
        history of training loss, training accuracy, validation loss, and
        validation accuracy.
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

            x    = batch[0].to(device)
            y    = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            loss.backward()
            optimizer.step()

            train_loss         += loss.data.item() * x.size(0)
            num_train_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_train_examples += x.shape[0]

        train_acc   = num_train_correct / num_train_examples
        train_loss  = train_loss / len(train_dl.dataset)


        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_loss       = 0.0
        num_val_correct  = 0
        num_val_examples = 0

        for batch in val_dl:

            x    = batch[0].to(device)
            y    = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            val_loss         += loss.data.item() * x.size(0)
            num_val_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_val_examples += y.shape[0]

        val_acc  = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_dl.dataset)


        print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
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

    print('Finished Training')

if __name__ == '__main__':
    main()
