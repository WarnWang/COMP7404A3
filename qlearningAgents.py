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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math

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
        if not actions:
            return 0.0

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
        action = None
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
        weights = self.getWeights()
        return self.featExtractor.getFeatures(state, action) * weights

    def get_feature_vector(self, state, action):
        """
        Use to get all the features
        :param state:
        :param action:
        :return: the feature vector
        """

        def get_normalize_data(data):
            return 1.0 / (1 + math.exp(-data))

        # fill featureVector
        feature_vector = util.Counter()
        feature_vector[CURRENT_SCORE] = state.getScore()
        if state.isWin():
            feature_vector[TERMINATED_OR_NOT] = 10
        elif state.isLose():
            feature_vector[TERMINATED_OR_NOT] = -10
        elif action is not None:

            # Generate next state
            next_state = state.generatePacmanSuccessor(action)

            # fill in score
            feature_vector[NEXT_SCORE] = next_state.getScore()

            if next_state.isWin():
                feature_vector[TERMINATED_OR_NOT] = 30
            elif next_state.isLose():
                feature_vector[TERMINATED_OR_NOT] = -30
            else:
                feature_vector[TERMINATED_OR_NOT] = 1
                feature_vector[VALID_ACTION_NUM] = len(next_state.getLegalPacmanActions())

                # Get some parameters
                pacman_pos = next_state.getPacmanPosition()
                ghost_pos = next_state.getGhostPositions()
                ghost_state = next_state.getGhostStates()
                food_pos = next_state.getFood()
                wall_pos = next_state.getWalls()

                # Fill nearest ghost distance and average ghost distance
                min_distance = float('inf')
                total_distance = 0
                total_ghost = 0
                nearest_scared_distance = None
                for i in range(len(ghost_pos)):
                    distance = util.manhattanDistance(pacman_pos, ghost_pos[i])
                    if ghost_state[i].scaredTimer > 0:
                        if nearest_scared_distance is None or nearest_scared_distance > distance:
                            nearest_scared_distance = distance
                    else:
                        if distance < min_distance:
                            min_distance = distance
                        total_distance += distance
                        total_ghost += 1

                if total_ghost:
                    feature_vector[AVERAGE_GHOST_DISTANCE] = -float(total_distance) / total_ghost
                    feature_vector[NEAREST_GHOST_DISTANCE] = -min_distance
                if nearest_scared_distance is not None and nearest_scared_distance > 0:
                    feature_vector[SCARED_GHOST_DISTANCE] = nearest_scared_distance

                # Find the nearest food position
                pos_to_explore = []
                explored_pos = set()
                pos_to_explore.append((pacman_pos, 0))
                distance = 0
                while pos_to_explore:
                    frontier = pos_to_explore.pop(0)
                    explored_pos.add(frontier[0])
                    if food_pos[frontier[0][0]][frontier[0][1]]:
                        distance = frontier[1]
                        break
                    else:
                        successor_pos = [(frontier[0][0] + 1, frontier[0][1]), (frontier[0][0] - 1, frontier[0][1]),
                                         (frontier[0][0], frontier[0][1] + 1), (frontier[0][0], frontier[0][1] - 1)]
                        for pos in successor_pos:
                            if pos[0] < 0 or pos[0] >= food_pos.width or pos[1] < 0 or pos[1] >= food_pos.height:
                                continue
                            if not wall_pos[pos[0]][pos[1]] and pos not in explored_pos:
                                pos_to_explore.append((pos, frontier[1] + 1))
                feature_vector[NEAREST_FOOD_DISTANCE] = -distance
                feature_vector[FOOD_NUM] = -len(food_pos.asList())

        for key in feature_vector:
            feature_vector[key] = get_normalize_data(feature_vector[key])

        return feature_vector

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        next_action = self.getAction(nextState)
        if next_action is None:
            return

        difference = reward + self.discount * self.getQValue(nextState, next_action) - self.getQValue(state, action)
        feature_vector = self.featExtractor.getFeatures(state, action)
        # print feature_vector

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
            print len(self.weights)
