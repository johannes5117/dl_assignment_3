from keras.models import load_model

### test the CNN if predictions are right

model = load_model('robobust.h5')
from utils     import Options
from simulator import Simulator
from transitionTable import TransitionTable
import numpy as np

opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                             opt.minibatch_size, opt.valid_size,
                             opt.states_fil, opt.labels_fil)
train_data = trans.get_train()
valid_data = trans.get_valid()
print(train_data[0].shape)
print(train_data[1].shape)


np_train_data = np.array(train_data[0])
training_data_x = np.reshape(np_train_data, (np_train_data.shape[0], 22, 25, 25))
training_data_x = np.rot90(training_data_x, axes=(1, 2))
training_data_x = np.rot90(training_data_x, axes=(2, 3))


for i in range(0,200):
    to_predict = i
    prediction = model.predict(training_data_x[to_predict:to_predict+1,:,:,:])
    print("Predicted: "+str(np.argmax(prediction))+ " Real: "+str(np.argmax(train_data[1][to_predict])))


    # debug output
    era = training_data_x[to_predict:to_predict + 1, :, :, :]
    print(era.shape)
    era[era > 240] = 4
    era[era > 100] = 3 # target location
    era[era > 50] = 2  # occupied
    era[era > 10] = 1  # robot position
                  # 0  = free
    sug = np.array(era, dtype=np.uint8)

    np.savetxt('f1.txt', sug[0, :, :, 21], '%i')
    if np.max(sug[0, :, :, 21] > 2):
        input("Hey")
    print("DGB")