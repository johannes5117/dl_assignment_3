# Simple Visual Planner Exercise
This folder contains the coded that you need to get started with the visual planner learning exercise.

# Overview
There are five main files that should be of interest for you:

  The file astar_demo.py contains a demo script that displays how an optimal A* planner would solve the planning problem this is roughly the performance that your agent can achieve in the end. You will train your agent to imitate this A* planner. Keep in mind though that the A* algorithm has full knowledge about the world and how transitions happen whereas your agent only gets a local view of the environment via an image.

  The get_data.py script generates some training data for you and stores by running the A* planner for a large number of steps, collecting the corresponding images and executed optimal actions, and saves the resulting data into a file.

  The train_agent.py script is the file you should adapt to train a neural network to predict the actions of the A* planner.
  
  The test_agent.py script is meant to be run after you have trained your agent and you should adapt it to test how well your agent is doing.

  The utils.py file contains some additional options that you might find useful NOTE: especially the disp_on variable might be interesting as it switches between running the scripts with and without the visualization on

Additionally you can change the map layour (or add new maps) by drawing a map in the format defined in maps.py.

# Results
#### Measured with following CNN:

Layer 1: 32 Filter, 3x3 Kernel, relu activation <br>
Layer 2: 64 Filter, 3x3 Kernel, relu activation <br>
MaxPoolingLayer: 2x2 Kernel <br>
Dropout: 25% <br>
FullyConnectedLayer: 128 Filter, relu activation <br>
Dropout: 50% <br>
Activation: softmax <br>
Optimizer: Adadelta (with accuracy metrics)

Trained on:
minibatch_size  = 32
n_minibatches   = 500
standard_window with 5x5

##### Historylength (h) Variation:
h=1:    Test accuracy: 0.93 , episod_solved/episod = 0.756 <br>
h=2:    Test accuracy: 0.94 , episod_solved/episod = 1 <br>
h=3:    Test accuracy: 0.953, episod_solved/episod = 1 <br>
h=5:    Test accuracy: 0.954, episod_solved/episod = 1 <br>
h=10:   Test accuracy: 0.978, episod_solved/episod = 1 <br>
h=20:   Test accuracy: 0.968, episod_solved/episod = 0.943

##### Partial Observation (po) Variation:
Tested with a history length of 4 

po=3:   Test accuracy: 0.88 , episod_solved/episod = 0.898 <br>
po=5:   Test accuracy: 0.974, episod_solved/episod = 1 <br>
po=7:   Doesn't work with the map since the border is only 2 thick the window would go beyond 

Observation: 
- A bigger window makes the scene the robo sees more identificable. <br> 

##### Changing the target position:
Even with different historylengths the agent has only a little chance to find the target (if it's on its way he'll make it).
We found it very funny how he held to the position he thought it should be and looked it up several times. <br>
We find that, this result is obvious, since we trained the agent on the A* for the old position. There is nothing in his logic to pursue the real position of the target. 

##### Changing the map
Changing the map:
 - a little (img2) bit (little bit = blocking some paths, adding new ones, but preserving main structure): preserves the chance that the robo finds his target. With a step limit of 75 he manages to find the target in around 65% of the trials. (We let the test_agent run several times and averaged)
 - major changes (img3) (major changes = rotating the whole map, keeping target position on old place): the agent is incapable of finding the target. In only 10% of trials he finds the target within 75 steps. 
 
 ![alt text](map_unchanged.png)
 
 img1: unchanged map
 
  ![alt text](map_slightly_changed.png)
 
 img2: map with minor changes
 
  ![alt text](map_major_changes.png)
 
 img3: map with major changes
 
 Conclusion: The agent generalizes badly. With small changes he can still sometimes (luckily) find the target. With major changes he has no chance.
 

#### Further ideas
With our model currently used our robo doesn't generalize at all on other maps then he was trained for. Maybe we are using the wrong kind of system for such a task. With a CNN the robo learns on the input horizon we give him. As the map stays the same it seems natural that he can easily manage to learn the commands. But with a new map the old sequences (horizons) aren't working anymore. 

We could think about a solution where the robo learns on different maps. The feature selection could then be completly different. E.g. instead of learning a sequence of input images, using the wall as help for navigation. 

There are several more possible ways to improve the generalization. E.g. we could train different on different maps and take a 	confidentially measurement for several predictions and let an arbiter chooce the highest or so.

We implemented the first idea of learning on several different maps. It didn't work. The result didn't generalize either. For further improvements we hope that in the next exercise we take a better suiting system than CNN's.