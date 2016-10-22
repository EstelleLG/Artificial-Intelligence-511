# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


    if successorGameState.isWin():
      return 1000000;
    x = newPos[0];
    y = newPos[1];

    foodNeighbor = 0;
    for yy in [y-1, y+1]:
      for xx in [x-1, x+1]:
        if newFood[xx][yy]:
          foodNeighbor += 5;

    foodDist = [util.manhattanDistance(food, newPos) for food in newFood.asList()]
    test = 1000./sum(foodDist)
    minFood = 50./min(foodDist);
    foodDist = test + minFood;

    ghostDist = 0;
    for ghostState in newGhostStates:
        ghostPos = ghostState.getPosition();
        dist = util.manhattanDistance(newPos, ghostPos);
        if dist == 0 :
            ghostDist = -100000;
        elif dist < ghostState.scaredTimer :
            ghostDist += 0.05*dist;
        elif dist < 4:
            ghostDist -= 1000./dist;
        else :
            ghostDist += 0.05*dist;

    eat = 0;
    currentFood = currentGameState.getFood();
    if currentFood[newPos[0]][newPos[1]]:
        eat = 100;


    "*** YOUR CODE HERE ***"
    return foodNeighbor+foodDist+ghostDist+eat;
    '''return successorGameState.getScore()'''

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def maxValue(self, gameState, level, agent):
      v = -float('Inf');
      for action in gameState.getLegalActions(agent):
          v = max(v, self.value(gameState.generateSuccessor(agent, action), level, 1));
      return v;

  def minValue(self, gameState, level, agent):
      nextAgent = agent+1;
      if nextAgent == gameState.getNumAgents():
          level += 1;
          nextAgent = 0;
      v = float('Inf');
      for action in gameState.getLegalActions(agent):
          v = min(v, self.value(gameState.generateSuccessor(agent, action), level, nextAgent))
      return v;

  def value(self, gameState, level, agent):
    if (level > self.depth):
        return self.evaluationFunction(gameState)
    if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

    if agent == 0:
        return self.maxValue(gameState, level, agent);
    if agent >= 1:
        return self.minValue(gameState, level, agent);


  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"

    legalActions = gameState.getLegalActions(0);
    mValue = -float('Inf')
    mAction = Directions.STOP;
    for action in legalActions:
        temp = self.value(gameState.generateSuccessor(0,action), 1, 1);
        if temp > mValue:
            mValue = temp;
            mAction = action;
    return mAction

    util.raiseNotDefined()



class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def maxValue(self, gameState, level, agent, alpha, beta):
      v = -float('Inf');
      for action in gameState.getLegalActions(agent):
          v = max(v, self.value(gameState.generateSuccessor(agent, action), level, 1, alpha, beta));
          if v >= beta:
              return v;
          alpha = max(alpha, v)
      return v;

  def minValue(self, gameState, level, agent, alpha, beta):
      nextAgent = agent+1;
      if nextAgent == gameState.getNumAgents():
          level += 1;
          nextAgent = 0;
      v = float('Inf');
      for action in gameState.getLegalActions(agent):
          v = min(v, self.value(gameState.generateSuccessor(agent, action), level, nextAgent, alpha, beta))
          if v <= alpha:
              return v;
          beta = min(beta, v);
      return v;

  def value(self, gameState, level, agent, alpha, beta):
    if (level > self.depth):
        return self.evaluationFunction(gameState)
    if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

    if agent == 0:
        return self.maxValue(gameState, level, agent, alpha, beta);
    if agent >= 1:
        return self.minValue(gameState, level, agent, alpha, beta);

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"

    legalActions = gameState.getLegalActions(0);
    mValue = -float('Inf')
    mAction = Directions.STOP;
    for action in legalActions:
        temp = self.value(gameState.generateSuccessor(0,action), 1, 1, -float('Inf'), float('Inf'));
        if temp > mValue:
            mValue = temp;
            mAction = action;

    return mAction

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def maxValue(self, gameState, level, agent):
      v = -float('Inf');
      for action in gameState.getLegalActions(agent):
          v = max(v, self.value(gameState.generateSuccessor(agent, action), level, 1));
      return v;

  def expValue(self, gameState, level, agent):
      nextAgent = agent+1;
      if nextAgent == gameState.getNumAgents():
          level += 1;
          nextAgent = 0;
      v = 0;
      legalActions = gameState.getLegalActions(agent);
      p = 1./len(legalActions);
      for action in gameState.getLegalActions(agent):
          v += p*self.value(gameState.generateSuccessor(agent, action), level, nextAgent)
      return v;

  def value(self, gameState, level, agent):
    if (level > self.depth):
        return self.evaluationFunction(gameState)
    if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

    if agent == 0:
        return self.maxValue(gameState, level, agent);
    if agent >= 1:
        return self.expValue(gameState, level, agent);



  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    legalActions = gameState.getLegalActions(0);
    mValue = -float('Inf')
    mAction = Directions.STOP;
    for action in legalActions:
        temp = self.value(gameState.generateSuccessor(0,action), 1, 1);
        if temp > mValue:
            mValue = temp;
            mAction = action;
    return mAction
    util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  food = currentGameState.getFood();
  foodList = food.asList();
  pacmanPos = currentGameState.getPacmanPosition();
  ghostStates = currentGameState.getGhostStates();

  if currentGameState.isWin():
      return float('inf');
  if currentGameState.isLose():
      return -float('inf');

  x = pacmanPos[0];
  y = pacmanPos[1];

  foodDist = [util.manhattanDistance(foodI, pacmanPos) for foodI in foodList]
  #test = 100./sum(foodDist)
  test = 0
  minFood = 10./min(foodDist);
  foodDist = test + minFood;

  minDist = float('inf');
  minState = None;
  for ghostState in ghostStates:
      ghostPos = ghostState.getPosition();
      dist = util.manhattanDistance(pacmanPos, ghostPos);
      if dist < minDist:
          minDist = dist;
          minState = ghostState;

  if minDist < minState.scaredTimer:
      test = 100./minDist
  elif minDist == 0:
      return -10000;
  else:
      test = -10./minDist


  cap = currentGameState.getCapsules();
  capDist = [util.manhattanDistance(capI, pacmanPos) for capI in cap];
  minCap = 0;
  if len(capDist) > 0:
      minCap = 30./min(capDist);


  return foodDist + currentGameState.getScore()+ minCap+test;
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
