# PREPROCESSING

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import model

# import tensorflow as tf
# from keras.preprocessing.image import img_to_array,load_img
#from tensorflow.keras import layers
#from tensorflow.keras.models import Sequential

import os
from bs4 import BeautifulSoup

# Loading data

def main():

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


    # for epoch in range(2):  # loop over the dataset multiple times

    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader, 0): # where trainloader = torch.utils.data.DataLoader(...)
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data

    #         # zero the parameter gradients
    #         optimizer.zero_grad()

    #         # forward + backward + optimize
    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels) # Classification Cross-Entropy loss 
    #         loss.backward()
    #         optimizer.step() # SGD with momentum optimizer

    #         # print statistics
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:    # print every 2000 mini-batches
    #             print('[%d, %5d] loss: %.3f' %
    #                 (epoch + 1, i + 1, running_loss / 2000))
    #             running_loss = 0.0

    print('Finished Training')