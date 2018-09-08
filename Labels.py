import os
import pandas as pd
import numpy as np
import cv2
import math
from keras.layers import Input
import tenserflow




train_images = os.listdir("train/")
test_images = os.listdir("test/")
df = pd.read_csv("train.csv")
df["imagepath"] = df["Image"].map(lambda x : "C:/Users/hima1993/Documents/3rd Year/Machine Learning/data/train/"+x)
ImageToLabelDict =  dict(zip(df.Image,df["Id"]))
df.head()

def Imageprocessing(imagepath,name,newfolder):
    img = cv2.imread(imagepath)
    height, width, channels = img.shape

    
    if channels != 1:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)


    img = np.atleast_3d(img)
    print("new0")
    print(img.shape)
    img = img.astype(np.float)
    
    if width > height:
        newsize = width
    else:
        newsize = height
    if newsize%2 == 1:
        newsize = newsize+1
    if newsize<896:
        newsize =896

    new_img = np.zeros((newsize,newsize))

    
    midh = int(newsize/2)
    whalf = int(math.floor(width/2))
    hhalf = int(math.floor(newsize/2))

    img = cv2.resize(img,((whalf)*2,(hhalf)*2))
    print("new")
    print(img.shape)
    
    new_img[(midh-hhalf):(midh+(hhalf)),\
    (midh-whalf):(midh+(whalf))] = img
   
    print("new1")
    print(new_img.shape)
    new_img = cv2.resize(new_img,(896,896))
    new_img = new_img.astype(np.float)
    new_img = (new_img-new_img.min())/(new_img.max()-new_img.min())
    new_img = new_img*255

    cv2.imwrite(newfolder+'/'+name,new_img)

for i in range(0,5):
    Imageprocessing(df.imagepath[i],df.Image[i],'test2')

input_img = Input(shape=(896,896,1))

x = Conv2D(16,(3,3),padding='same')(input_img)
x=BatchNormalization()(x)
x=Activation('relu')()
x = Conv2D(16,(3,3),padding='same')(x)
x=BatchNormalization()(x)
x=Activation('relu')(x)
x=MaxPolling2D((2,2),padding='same')(x)

x=Conv2D(8,(3,3),padding='same')(x)
x=BatchNormalization()(x)
x=Activation('relu')(x)
x = Conv2D(3,(3,3),padding='same')(x)
x=BatchNormalization()(x)
x=Activation('relu')(x)
encoded = MaxPooling2D((2,2),padding='same')(x)

x=Conv2D(8,(3,3),padding='same')(encoded)
x=BatchNormalization()(x)
x=Activation('relu')(x)
x = Conv2D(8,(3,3),padding='same')(x)
x=BatchNormalization()(x)
x=Activation('relu')(x)
x=UpSampling2D((2,2))(x)

x=Conv2D(16,(3,3),padding='same')(encoded)
x=BatchNormalization()(x)
x=Activation('relu')(x)
x = Conv2D(16,(3,3),padding='same')(x)
x=BatchNormalization()(x)
x=Activation('relu')(x)
x=UpSampling2D((2,2))(x)
decoded = Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)










