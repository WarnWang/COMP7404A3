# qlearningAgents.py
# ------------------
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


import random

import util
from featureExtractors import *

from learningAgents import ReinforcementAgent

NEXT_SCORE = "score"
NEAREST_GHOST_DISTANCE = 'nearest_ghost'
AVERAGE_GHOST_DISTANCE = 'avg_ghost'
NEAREST_FOOD_DISTANCE = 'nearest_food'
FOOD_NUM = 'food_num'
VALID_ACTION_NUM = 'valid_action_num'
SCARED_GHOST_DISTANCE = 'ghost_scared_distance'
TERMINATED_OR_NOT = "is_terminated"
CURRENT_SCORE = "last_score"

WEIGHT_FEATURES = [NEXT_SCORE, NEAREST_FOOD_DISTANCE, NEAREST_GHOST_DISTANCE, AVERAGE_GHOST_DISTANCE, FOOD_NUM,
                   VALID_ACTION_NUM, SCARED_GHOST_DISTANCE, TERMINATED_OR_NOT, CURRENT_SCORE]


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.q_values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if (state, action) not in self.q_values:
            self.q_values[(state, action)] = 0.0
            return 0.0
        else:
            return self.q_values[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)

        # No valid actions
        if not actions:
            return 0.0

        # As we have handled the situation about no valid actions, so we don't need to think about it again here.
        max_q_value = float('-inf')
        for action in actions:
            q_value = self.getQValue(state, action)
            max_q_value = max(max_q_value, q_value)
        return max_q_value

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        max_value = float('-inf')
        best_action = []

        for action in actions:
            q_value = self.getQValue(state, action)
            if q_value > max_value:
                max_value = q_value
                best_action = [action]

            # As float number is hard to compare, if the difference between the two float is less than 1e-5, I will
            # regard these two float as same number
            elif abs(q_value - max_value) < 1e-5:
                best_action.append(action)

        return random.choice(best_action) if best_action else None

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        "*** YOUR CODE HERE ***"
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        sample_value = reward + self.discount * self.computeValueFromQValues(nextState)
        self.q_values[(state, action)] = (1 - self.alpha) * self.q_values[(state, action)] \
                                         + self.alpha * sample_value

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        feature_vector = self.featExtractor.getFeatures(state, action)
        return self.getWeights() * feature_vector

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        next_action = self.getPolicy(nextState)

        # Some featExtractor can handle none action, but some are not. so I add one try here to handle such case.
        try:
            next_q_value = self.getQValue(nextState, next_action)
        except Exception:
            next_q_value = 0

        # Calculate difference and update weights
        difference = reward + self.discount * next_q_value - self.getQValue(state, action)
        feature_vector = self.featExtractor.getFeatures(state, action)
        for key in feature_vector:
            self.weights[key] += self.alpha * difference * feature_vector[key]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
