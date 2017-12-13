import keras
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

from simulator import Simulator
from transitionTable import TransitionTable
# custom modules
from utils import Options

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

historyLength = opt.hist_len
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



np_train_data = np.array(train_data[0])
print(np_train_data.shape)
training_data_x = np.reshape(np_train_data, (
np_train_data.shape[0], historyLength, opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz))
training_data_x = np.rot90(training_data_x, axes=(1, 2))
training_data_x = np.rot90(training_data_x, axes=(2, 3))

np_valid_data = np.array(valid_data[0])
validation_data_x = np.reshape(np_valid_data, (
np_valid_data.shape[0], historyLength, opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz))
validation_data_x = np.rot90(validation_data_x, axes=(1, 2))
validation_data_x = np.rot90(validation_data_x, axes=(2, 3))

validation_data_y = valid_data[1]

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
epochs = 30

x_train = training_data_x[0:np_train_data.shape[0] - np.math.floor(opt.n_minibatches * 0.2), :, :, :]
x_test = training_data_x[np_train_data.shape[0] - np.math.floor(opt.n_minibatches * 0.2):np_train_data.shape[0], :, :,
         :]
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = train_data[1][0:np_train_data.shape[0] - np.math.floor(opt.n_minibatches * 0.2)]
y_test = train_data[1][np_train_data.shape[0] - np.math.floor(opt.n_minibatches * 0.2):np_train_data.shape[0]]


print(y_train.shape)
print("Historylength: " + str(historyLength))
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz, historyLength)))
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
score = model.evaluate(validation_data_x, validation_data_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('robobust.h5')


