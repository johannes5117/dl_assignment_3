import numpy as np
import tensorflow as tf

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# custom modules
from utils     import Options
from simulator import Simulator
from transitionTable import TransitionTable


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this script assumes you did generate your data with the get_data.py script
# you are of course allowed to change it and generate data here but if you
# want this to work out of the box first run get_data.py
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                             opt.minibatch_size, opt.valid_size,
                             opt.states_fil, opt.labels_fil)

historyLength = 4
# 1. train
######################################
# TODO implement your training here!
# you can get the full data from the transition table like this:
#
# # both train_data and valid_data contain tupes of images and labels
train_data = trans.get_train()
valid_data = trans.get_valid()
print(train_data[0].shape)
print(train_data[1].shape)

xdata = np.array(train_data[0])


newShape = np.reshape(xdata, (xdata.shape[0],4,25,25))
print(newShape.shape)
newShape = np.rot90(newShape, axes=(1,2))
newShape = np.rot90(newShape, axes=(2,3))
print(newShape.shape)

#newShape = newShape[:,0::5,0::5,:]

newShape[newShape>50] = 2
newShape[newShape>10] = 1

#print(newShape[0].shape)
#newShape = newShape[0][0::5,0::5,:][:,:,0]
print(newShape[0,:,:,0])
print("")
print(newShape[0,:,:,1])
print("")
print(newShape[0,:,:,2])
print("")
print(newShape[0,:,:,3])




# 
# alternatively you can get one random mini batch line this
#
# for i in range(number_of_batches):
#     x, y = trans.sample_minibatch()
# Hint: to ease loading your model later create a model.py file
# where you define your network configuration
######################################

batch_size = 128
num_classes = 5
epochs = 12



x_train = newShape[0:xdata.shape[0]-2000,:,:,:]
x_test =  newShape[xdata.shape[0]-2000:xdata.shape[0],:,:,:]
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = train_data[1][0:xdata.shape[0]-2000]
y_test =  train_data[1][xdata.shape[0]-2000:xdata.shape[0]]
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_train.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(25,25,4)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# 2. save your trained model


