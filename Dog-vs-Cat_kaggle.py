from keras import applications
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
import numpy as np
import pandas as pd
import os
from PIL import Image
import tensorflow as tf
import keras
import glob 
from sklearn.model_selection import train_test_split


def convert_image_to_data(image, WIDTH, HEIGHT):
    image_resized = Image.open(image).resize((WIDTH, HEIGHT))
    image_array = np.array(image_resized)
    return image_array



dog_files=glob.glob("ashiyashi/datasets/sent-data/1/dog*")
cat_files = glob.glob("ashiyashi/datasets/sent-data/1/cat*")
print cat_files.shape


def creating_test_train_datset(WIDTH,HEIGHT):
    #cat=glob.glob("/home/ashiya/Downloads/dog_vs_cat/train//cat.*")
    #dog=glob.glob("/home/ashiya/Downloads/dog_vs_cat/train/dog*")
    
    cat_list=[convert_image_to_data(i, WIDTH, HEIGHT) for i in cat_files]
    dog_list=[convert_image_to_data(i,WIDTH,HEIGHT) for i in dog_files]
    
    y_cat_list=np.zeros(len(cat_list))
    y_dog_list=np.zeros(len(dog_list))
    x=np.concatenate([cat_list,dog_list])
    y=np.concatenate([y_cat_list,y_dog_list])
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)
    return X_train, X_test, y_train, y_test


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

num_classes = 2
X_train,X_test,y_train,y_test=creating_test_train_datset(224,224)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print X_train.shape[1:]


batch_size = 32

epochs = 100


model= Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


#adam
opt = keras.optimizers.adam(lr=0.0001,beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.)
#rmsprop
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

model.summary()

model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test))
test_files=glob.glob("/home/ashiya/Downloads/dog_vs_cat/test1/*")
test_image_list=[convert_image_to_data(i,HEIGHT,WIDTH) for i in test_files]

y_test_predict = model.predict(test_image_list)



