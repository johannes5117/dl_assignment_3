import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)

# TODO: load your agent
# Hint: If using standard tensorflow api it helps to write your own model.py
# file with the network configuration, including a function model.load().
# You can use saver = tf.train.Saver() and saver.restore(sess, filename_cpkt)
from keras.models import load_model
model = load_model('robobust.h5')
historyLength = 22


# 1. control loop
if opt.disp_on:
    win_all = None
    win_pob = None
epi_step = 0    # #steps in current episode
nepisodes = 0   # total #episodes executed
nepisodes_solved = 0
action = 0     # action to take given by the network

# start a new game
state = sim.newGame(opt.tgt_y, opt.tgt_x)

# history with n Images
history = []

for step in range(opt.eval_steps):

    # check if episode ended
    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        if state.terminal:
            nepisodes_solved += 1
        # start a new game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
    else:
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO: here you would let your agent take its action
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Hint: get the image using rgb2gray(state.pob), append latest image to a history 
        # this just gets a random action
        if(len(history) == 0):
            for i in range(0, historyLength):
                history.append(rgb2gray(state.pob))

        rgb2gray(state.pob).reshape(opt.state_siz)

        # Replace the last image with the new one to obtain a consistent history
        # TODO: wrap python array into numpy array.
        history.pop(0)
        history.append(rgb2gray(state.pob))
        stack = np.rot90(np.rot90(np.array(history[0])))
        for i in range(1,historyLength):
            stack = np.dstack((stack, np.rot90(np.rot90(np.array(history[i])))))


        # stack[stack > 50]  = 2
        # stack[stack > 10]  = 1
        # stack = np.array(stack, dtype=np.uint8)
        #
        #
        # np.savetxt('f1.txt',stack[:,:,0], '%i')
        # np.savetxt('f2.txt',stack[:,:,1], '%i')
        # np.savetxt('f3.txt',stack[:,:,2], '%i')
        # np.savetxt('f4.txt',stack[:,:,3], '%i')


        newS = np.zeros((1,25,25,historyLength))

        newS[0] = stack
        action = model.predict(newS)
        #action = randrange(opt.act_num)
        action = np.argmax(action)
        state = sim.step(action)
        epi_step += 1


    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        if state.terminal:
            nepisodes_solved += 1
            print(nepisodes_solved)
        else:
            print("knock out")
        # start a new game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)

    if opt.disp_on:
        if win_all is None:
            plt.subplot(121)
            win_all = plt.imshow(state.screen)
            plt.subplot(122)
            win_pob = plt.imshow(state.pob)
        else:
            win_all.set_data(state.screen)
            win_pob.set_data(state.pob)
        plt.pause(opt.disp_interval)
        plt.draw()

# 2. calculate statistics
print(float(nepisodes_solved) / float(nepisodes))
# 3. TODO perhaps  do some additional analysis
