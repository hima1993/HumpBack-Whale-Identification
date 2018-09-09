import os
import pandas as pd
import numpy as np
import cv2
import math
import tensorflow
import keras
import keras.utils
from keras import utils
from keras.layers.convolutional import Conv2D
from keras.layers import Input, Dense,Activation,MaxPooling2D,UpSampling2D
from keras.layers.normalization import BatchNormalization
from tensorflow.python.client import device_lib
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard


train_images = os.listdir("train/")
test_images = os.listdir("test/")
df = pd.read_csv("train.csv")
df["imagepath"] = df["Image"].map(lambda x : "C:/Users/hima1993/Documents/3rd Year/Machine Learning/data/train/"+x)
ImageToLabelDict =  dict(zip(df.Image,df["Id"]))
df.head()

def Imageprocessing(imagepath,name,newfolder):
    img = cv2.imread(imagepath)
    
    height, width, channels = img.shape
    heightnew = height
    widthnew = width
    channelnew = channels
    print(heightnew)
    
    if channels != 1:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        


   
    
    print("here image shape")
    print(np.newaxis)
   
    img = img.astype(np.float)

    
   
    
    if width > height:
        newsize = width
    else:
        newsize = height
    if newsize%2 == 1:
        newsize = newsize+1
    if newsize<896:
        newsize =896

    new_img = np.zeros((newsize,newsize,3))

    midh = int(newsize/2)
    whalf = int(math.floor(width/2))
    hhalf = int(math.floor(newsize/2))

    img = cv2.resize(img,((whalf)*2,(hhalf)*2))
    img = img[:, :, np.newaxis]
    print(img.shape)
    
    new_img[(midh-hhalf):(midh+(hhalf)),\
            (midh-whalf):(midh+(whalf)),:] = img
   
    print(new_img.shape)
    new_img = cv2.resize(new_img,(896,896))
    new_img = new_img.astype(np.float)
   
    new_img = (new_img-new_img.min())/(new_img.max()-new_img.min())
    new_img = new_img*255

    cv2.imwrite(newfolder+'/'+name,new_img)

for i in range(0,10):
    Imageprocessing(df.imagepath[i],df.Image[i],'test2')

input_img = Input(shape=(896,896,1))

x = Conv2D(16,(3,3),padding='same')(input_img)
x=BatchNormalization()(x)
x=Activation('relu')(x)
x = Conv2D(16,(3,3),padding='same')(x)
x=BatchNormalization()(x)
x=Activation('relu')(x)
x=MaxPooling2D((2,2),padding='same')(x)

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
##def get_available_gpus():
##   local_device_protos = device_lib.list_local_devices()
##   return [x.name for x in local_device_protos if x.device_type    == 'GPU']
##num_gpu = len(get_available_gpus())
##print("...................")
##print(num_gpu)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
autoencoder = Model(input_img,decoded)
ot = keras.optimizers.Adam(lr=0.5,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.95,amsgrad = False)
epochsnum =20
autoencoder = multi_gpu_model(autoencoder,gpus = 2)
autoencoder.compile(optimizer = ot,loss="mean_squared_error")
autoencoder.summary()

batch_size = 64
seednum = 1

train_gen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 15,
    width_shift_range = .15,
    height_shift_range = .15,
    horizontal_flip=True)

trainD = train_gen.flow_from_directory(
    'AutoEncode/train/',
    class_mode="input",
    target_size = (896,896),
    color_mode = 'grayscale',
    batch_size = batch_size,
    seed=seednum)

test_datagen=ImageDataGenerator(rescale=1./255)

testD = test_datagen.flow_from_directory(
    'AutoEncode/test/',
    target_size = (896,896),
    batch_size = batch_size,
    color_mode = 'grayscale',
    class_mode="input",
    seed=seednum+1)

    
autoencoder.fit_generator(trainD,
                          epochs = epochsnum,
                          verbose=1,
                          shuffle=True,
                          validation_data=testD,
                          steps_per_epoch=9792//batch_size,
                          validation_steps=9792//batch_size,
                          callbacks=[TensorBoard(log_dir='/tmp/traine5')])







