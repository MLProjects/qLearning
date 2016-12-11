from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
          
class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent
    
    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update
      
    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.gamma (discount rate)
    
    Functions you should use
      - self.getLegalActions(state) 
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    "*** YOUR CODE HERE ***"
    self.Q = util.Counter()
  
  def getQValue(self, state, action):
    """
      Returns Q(state,action)    
      Should return 0.0 if we never seen
      a state or (state,action) tuple 
    """
    "*** YOUR CODE HERE ***"
    return self.Q[(state, action)]
    util.raiseNotDefined()
  
    
  def getValue(self, state):
    """
      Returns max_action Q(state,action)        
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    
    legalActions = self.getLegalActions(state)
    maxQValue = -1000.0
    max_action = 0
    if legalActions:
        for action in legalActions:
            if maxQValue < self.getQValue(state, action):
                maxQValue = self.getQValue(state, action)
                max_action = action
    else:
        maxQValue = 0.0

    return ( max_action, maxQValue )
    util.raiseNotDefined()
    
  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    "*** YOUR CODE HERE ***"
    max_action, maxQValue = self.getValue(state)
    
    if max_action == 0:
        return None

    #print "Best action is: ", max_action
    return max_action
    util.raiseNotDefined()
    
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
    takeRandom = util.flipCoin(self.epsilon)
    policyAction = self.getPolicy(state)
    action = None

    if legalActions:
        if takeRandom:
            action = random.choice(legalActions)
        else:
            action = policyAction
    
    #print "Policy anction is " , policyAction, "legal actions ", legalActions, "we take", action
    
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
    bestNexAct, nStateUtil = self.getValue(nextState)
    self.Q[(state,action)] = ( 1 - self.alpha)*self.Q[(state,action)] + self.alpha*(reward + self.gamma*nStateUtil)
    #print( "Value of Q is:", self.Q[(state,action)],"for state:", state, "and action:", action ) 
    #print( "The size of Q is: ", len(self.Q) )
    #util.raiseNotDefined()
    
class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"
  
  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
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
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
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

    # You might want to initialize weights here.
    self.weight = util.Counter()
    "*** YOUR CODE HERE ***"
    
  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    
    "*** YOUR CODE HERE ***"
    
    featureVector = self.featExtractor.getFeatures(state, action)
    #print "Feature Vector is", featureVector
    #print "State is: ", state
    #print "Action is: ", action
    #print "Weights are: ", self.weight
    sumQ = 0
    for feature in featureVector:
        #print( "Feature is: ", featureVector[feature] )
        sumQ += featureVector[feature] * self.weight[(feature)]
    return sumQ
    util.raiseNotDefined()
    
  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition  
    """
    "*** YOUR CODE HERE ***"
    featureVector = self.featExtractor.getFeatures(state, action)
    #print "Debug in", state,"Debug inner", action, "Debug out"
    bestNexAct, nStateUtil = self.getValue(nextState)
    reward + self.gamma*nStateUtil
    correction = reward + self.gamma*nStateUtil - self.getQValue(state, action)
    
    for feature in featureVector:
        self.weight[(feature)] += self.alpha*correction*featureVector[feature]
        
    #print "Weights are: ", self.weight
    #util.raiseNotDefined()
    
  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)
    
    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      print "Weights are: ", self.weight
      pass
