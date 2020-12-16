# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os
from bs4 import BeautifulSoup

# Loading data
img_folder = "C:/Users/profi/Downloads/Face_Mask_Dataset_(from Kaggle)/images"
annot_folder = "C:/Users/profi/Downloads/Face_Mask_Dataset_(from Kaggle)/annotations"

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

img_name

# One Hot Encoding label data
labels = pd.get_dummies(label)
print(labels.head())

# Our target classes
classes = list(labels.columns)
print(classes)

# Loading Images and converting them to pixel array
data, target = [],[]
img_h, img_w = 256, 256

for i in range(len(img_name)):
    name = img_name[i]
    path = img_folder + name
    
    image = load_img(path, target_size = (img_h, img_w))
    image = img_to_array(image)
    data.append(image)
    target.append(tuple(labels.iloc[i,:]))

# Converting list to array
data=np.array(data,dtype="float32")/255.0
target=np.array(target,dtype="float32")

# Visualizing few images randomly
plt.figure(figsize=(10, 10))
for i,j in enumerate(np.random.randint(1, 500, 9, dtype=int)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(data[j])
    plt.title(label[j])
    plt.axis("off")

# Shape of data and target
data.shape, target.shape

# Splitting into train and test data
train_img,test_img,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=20)

print("Train shapes : ",(train_img.shape, y_train.shape))
print("Test shapes : ",(test_img.shape, y_test.shape))
