

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
import numpy as np
import pandas as pd
import os
from PIL import Image
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
#import seaborn as sns
import glob
%matplotlib inline 
from sklearn.model_selection import train_test_split


def convert_image_to_data(image, WIDTH, HEIGHT):
    image_resized = Image.open(image).resize((WIDTH, HEIGHT))
    image_array = np.array(image_resized)
    return image_array

dog_files=glob.glob("/home/aries/Desktop/train/dog*")
#dog_files=dog_files[:2000]
cat_files = glob.glob("/home/aries/Desktop/train/cat*")
#cat_files=cat_files[:2000]


width=224
height=224
epoch=100
batch_size=128


def creating_test_train_datset(WIDTH,HEIGHT):
    #cat=glob.glob("/home/ashiya/Downloads/dog_vs_cat/train//cat.*")
    #dog=glob.glob("/home/ashiya/Downloads/dog_vs_cat/train/dog*")
    
    cat_list=[convert_image_to_data(i, WIDTH, HEIGHT) for i in cat_files]
    dog_list=[convert_image_to_data(i,WIDTH,HEIGHT) for i in dog_files]
    
    y_cat_list=np.zeros(len(cat_list))
    y_dog_list=np.ones(len(dog_list))
    x=np.concatenate([cat_list,dog_list])
    y=np.concatenate([y_cat_list,y_dog_list])
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)
    return X_train, X_test, y_train, y_test


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

#divide in categories by giving itn numbers like 0,1,2
num_classes = 2
X_train,X_test,y_train,y_test=creating_test_train_datset(224,224)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print X_train.shape[1:]


#include top false karne se hati hai fully connected layer
model=keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',input_shape=(height,width,3))

model.summary()


for layer in model.layers[:3]:
    layer.trainable = False



#Adding custom Layers 
x = model.output
x = Flatten()(x)

x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(510, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

model = Model(inputs=model.input, outputs=predictions)
#opt = keras.optimizers.adam(lr=0.0001,beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.)
#model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
model.summary()

epochs=10
batch_size=128


#Adding custom Layers 
#x = model.output
#x = Flatten()(x)
#x = Dense(1024, activation="relu")(x)
#x = Dropout(0.5)(x)
#x = Dense(1024, activation="relu")(x)
#predictions = Dense(16, activation="softmax")(x)
# creating the final model 
opt = keras.optimizers.adam(lr=0.0001,beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.)
#rmsprop
#    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test))


test_files=glob.glob("/home/ashiya/Downloads/dog_vs_cat/test1/*")
test_image_list=[convert_image_to_data(i,HEIGHT,WIDTH) for i in test_files]
y_test_predict = model.predict(test_image_list)
df = pd.DataFrame(y_test)
df.to_csv("file_path.csv")