import numpy as np
import random
from collections import defaultdict

class Agent:

    def __init__(self, nA=4, alpha=0.01, gamma=1.0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.gamma = gamma

    def getPolicyFromQ(self, Q_s, epsilon):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] += (1.0 - epsilon)
        return policy_s
    
    def select_action(self, state, epsilon):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy = self.getPolicyFromQ(self.Q[state], epsilon)
        return np.random.choice(np.arange(self.nA), p=policy)

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
        self.Q[state][action] += self.alpha*(reward + (self.gamma*np.max(self.Q[next_state])) - self.Q[state][action])

