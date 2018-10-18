# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Implementation

Since the action space was discreet, I have used DQN algorithm with the below mentioned hyperparameters. A NN with two hidden layers, both containing 64 nodes, is used for creating the Q network and the implementation is in `model.py`. The overall initialisation of network, learning logic and implementation of experience replay is done in `agent.py`. The epsilon updation logic, for epsilon greedy exploration approach is directly implemented in the notebook and is passed as an argument to the agent.

- Replay buffer size: int(1e5)
- Batch size: 64
- Discount factor TAU: 0.99
- Soft update: 1e-3
- Learning Rate: 3e-4

Just to train my agent more, I kept the target score as 15.0 and was able to reach the threshold 13.0 in less than 500 episodes. 

```
Episode 400	Average Score: 10.44
Episode 500	Average Score: 13.24
Episode 600	Average Score: 13.70
Episode 675	Average Score: 15.01
```

For future improvements, I can explore Double DQN, Prioritized Experience Replay and Dueling DQN. Further trying out the optional part of learning directly from pixels also sounds interesting and challenging.

### Getting Started

1. Follow the [instructions](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) to install Unity ML-Agents. 

2. Navigate to the `p1_navigation/` folder, and run the command below to obtain a few more packages.
  ```
  pip3 install -r requirements.txt
  ```

3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

4. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  To use a Jupyter notebook, run the following command from the `p1_navigation/` folder:
```
jupyter notebook
```
and open `Navigation.ipynb` from the list of files.  Alternatively, you may prefer to work with the [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/) interface.  To do this, run this command instead:
```
jupyter-lab
```
