# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:46:03 2020

@author: subhra
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.applications.vgg16 import decode_predictions
from imutils import paths
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from os import sys, listdir


root_directory = 'E:\Subra\Data'

train_data=[]
test_data=[]
train_label=[]
test_label=[]

img_row,img_col = 224,224

nb_classes = 4

#creationo of classes based on the directory names
data_type =["train","test"]
folder_names=["Driving_License","Pancard","Passport","Voter_ID"]
label_mapping={"Driving_License":0,"Pancard":1,"Passport":2,"Voter_ID":3}


#image processing 
def process_img(dataset,folder,filename): 
    file_path = root_directory+"/"+dataset+"/"+folder+"/"+filename
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(224,224))
    return image 
    


#appending pictures and labels
for dataset in data_type:
    for folder in folder_names:
        for filename in listdir(root_directory+"/"+dataset+"/"+folder+"/"):
            if dataset == "train":
                train_data.append(process_img(dataset,folder,filename))
                train_label.append(label_mapping[folder])
            else:
                test_data.append(process_img(dataset,folder,filename))
                test_label.append(label_mapping[folder])

# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 255]

#normalising the image , so the value can not be less than 1 or greater than 1
train_data = np.asarray(train_data) / 255.0
train_label = np.asarray(train_label)

test_data = np.asarray(test_data) / 255.0
test_label = np.asarray(test_label)

#reshaing the images using NumPy
trainX = train_data.reshape(train_data.shape[0], img_row, img_col, 3)
testX = test_data.reshape(test_data.shape[0], img_row, img_col, 3)

trainY = np_utils.to_categorical(train_label, nb_classes)
testY = np_utils.to_categorical(test_label, nb_classes)


# initialize the training data augmentation object
trainAug = ImageDataGenerator(
 	rotation_range=15,
 	fill_mode="nearest")


# load the VGG16 network, ensuring the head FC layer sets are left
# off
baseModel = VGG16(weights="imagenet", include_top=False,
 	input_tensor=Input(shape=(224, 224, 3)))


# construct the head of the model that will be placed on top of the
# the base model baswically we are replacing VGG's dense layer with our 
# customized model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(nb_classes, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
 	layer.trainable = False

INIT_LR = 1e-3 #0.001
EPOCHS = 2 #one forward pass and one backward pass of all the training examples
BS = 8  #smaller batsize require less memory , trainn faster also every
        #epoch will be equally distributed

#compile our model
print("[INFO] compiling model...")
#opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer='rmsprop',
 	metrics=["accuracy"])


# train the head of the network
print("[INFO] training head...")
H = model.fit_generator(
 	trainAug.flow(trainX, trainY, batch_size=BS),
 	steps_per_epoch=len(trainX) // BS,
 	validation_data=(testX, testY),
 	validation_steps=len(testX) // BS,
 	epochs=EPOCHS)


# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)


target_names=["class 0(Driving_License)","class 1(Pancard)","class 2(Passport)","class 3(Voter_ID)"]

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
 	target_names=target_names))


# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(cm))
#acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")


model_json = model.to_json()
with open("doc_classification", "w") as json_file:
    json_file.write(model_json)
# serialize the model to disk
print("[INFO] saving COVID-19 detector model...")
model.save("doc_classification.h5")
model.save_weights('doc_classification_weights.h5')

#pan_new_pixel
#Pancard_Distance_Damaged #wrong prediction
#Passport_1_blur
#Passport_1_shear
#Passport_2_greyscale
#test
#pan_greyscale
#Passport_2_blur_grey
#Passport_1_grey

imagePath_1 = os.path.normpath('D:/Office_validation/Passport_1_grey.jpg')
label_1=imagePath_1.split(os.sep)[-2]
image_pred = cv2.imread(imagePath_1)
image_pred = cv2.cvtColor(image_pred, cv2.COLOR_BGR2RGB)
img_pred = cv2.resize(image_pred, (224, 224))
img_pred = np.array(img_pred) / 255.0
rslt = model.predict(img_pred.reshape(1,224,224,3))
rslt = rslt.argmax(axis=1)[0]
print(rslt)
#=====================================================================================================
#target_names=["class 0(Driving_License)","class 1(Pancard)","class 2(Passport)","class 3(Voter_ID)"]
#====================================================================================================


