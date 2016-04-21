# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util

from learningAgents import ValueEstimationAgent

WEST = 'west'
NORTH = 'north'
SOUTH = 'south'
EAST = 'east'
EXIT = 'exit'


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        ValueEstimationAgent.__init__(self, gamma=discount, numTraining=iterations)
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        all_states = self.mdp.getStates()

        # Init all the state values, the first iteration
        # terminal_state = []
        # for state in all_states:
        #     if self.mdp.isTerminal(state=state):
        #         x, y = state
        #         self.values[state] = self.mdp.grid[x][y]
        #         terminal_state.append(state)
        #     else:
        #         self.values[state] = 0.0

        # Do the calculation
        for i in range(0, iterations):
            new_count = self.values.copy()
            for state in all_states:
                action = self.computeActionFromValues(state=state)
                if action is not None:
                    new_count[state] = self.computeQValueFromValues(state, action)
            self.values = new_count

        # Set the grid to the calculated values
        # for state in self.values:
        #     x, y = state
        #     self.mdp.grid[x][y] = self.values[state]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        tran_states_probability = self.mdp.getTransitionStatesAndProbs(state=state, action=action)
        q_value = 0
        for next_state, probability in tran_states_probability:
            reward = self.mdp.getReward(state=state, action=action, nextState=next_state)
            q_value += probability * (reward + self.discount * self.getValue(next_state))
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        best_action = None
        best_q_value = float('-inf')
        for action in actions:
            q_value = self.computeQValueFromValues(state=state, action=action)
            if q_value > best_q_value:
                best_action = action
                best_q_value = q_value
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
