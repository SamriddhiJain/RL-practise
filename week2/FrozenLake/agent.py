import numpy as np
import random
from collections import defaultdict

class Agent:

    def __init__(self, nS = 16, nA = 4, alpha = 0.01, gamma = 1.0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nS = nS
        self.nA = nA
        self.Q = np.zeros((self.nS, self.nA))
        self.alpha = alpha
        self.gamma = gamma
    
    def select_eps_greedy_action(self, state, epsilon):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if np.random.uniform(0,1) < epsilon:
            return np.random.randint(0, self.nA)
        else:
            return np.argmax(self.Q[state,:])

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] += self.alpha*(reward + (self.gamma*np.max(self.Q[next_state,:])) - self.Q[state][action])

