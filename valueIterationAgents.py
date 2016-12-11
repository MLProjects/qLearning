import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
    self.policy = util.Counter()
    self.Q = util.Counter()
    
    # Fill in Q
    self.states = mdp.getStates()
    for iterator in range(0,self.iterations):
        for state in self.states:
            possibleActions = mdp.getPossibleActions(state) #Return list of possible actions from 'state'.
            for action in  possibleActions:
                probabilities = mdp.getTransitionStatesAndProbs(state, action) # Returns list of (nextState, prob) pairs
                self.Q[(state,action)] = 0
                for nextState, prob in probabilities:
                    #print('For state', state, 'and action', action, 'next state', nextState, 'and prob', prob  )
                    # def getReward(self, state, action, nextState):
                    reward = mdp.getReward(state, action, nextState)
                    #print('For state', state, 'action', action, 'and next state', nextState, 'reward is', reward)
                    self.Q[(state,action)] += prob * (reward + discount * self.values[nextState]) # Adding all the time
            max_Q = -1000
            for action in possibleActions: 
                if(self.Q[(state,action)] > max_Q):
                    max_Q = self.Q[(state,action)] 
                    self.values[state] = self.Q[(state,action)]
                    self.policy[state] = action

     
    "*** YOUR CODE HERE ***"
    
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    return self.Q[(state, action)]
    util.raiseNotDefined()

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    return self.policy[state]
    util.raiseNotDefined()

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
