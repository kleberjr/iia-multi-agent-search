# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

PACMAN_IDX = 0
INITIAL_DEPTH = 0
INFINITY = float("inf")

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"
        foodPositions = newFood.asList()
        
        # If there is no more food left, we won!
        if (len(foodPositions) == 0):
            return 100000000
        
        # Calculates the absolute distance from pacman to all ghosts
        ghostPositions = successorGameState.getGhostPositions()
        distanceFromGhosts = [manhattanDistance(position, newPos) for position in ghostPositions]

        # Don't let pacman get too close to the ghosts or it could end up trapped
        for distance in distanceFromGhosts:
            if (distance < 2):
                return -100000000
        
        # Calculates the absolute distance from pacman to all food dots
        distanceFromNewFood = [manhattanDistance(position, newPos) for position in foodPositions]

        # This number will increase as pacman gets closer to food dots
        closerToFoodDots = 1000/sum(distanceFromNewFood)
        # This number will increase as food runs out
        lessFoodLeft = 10000/len(distanceFromNewFood)

        # As food runs out and/or pacman gets closer to food dots, the score (probability of success) is higher
        return closerToFoodDots + lessFoodLeft

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(currentDepth, agentIdx, gameState):
            # Increase depth if all agents finished their turn
            if agentIdx >= gameState.getNumAgents():
                agentIdx = PACMAN_IDX
                currentDepth += 1
            
            # Reached max depth
            if currentDepth == self.depth:
                return None, self.evaluationFunction(gameState)
            
            highestScore, bestAction = None, None
            legalActions = gameState.getLegalActions(agentIdx)

            # Pacman's turn
            if agentIdx == 0:
                for action in legalActions: 
                    nextGameState = gameState.generateSuccessor(agentIdx, action)
                    
                    # Get the minimax score for the next agent
                    result = minimax(currentDepth, agentIdx + 1, nextGameState)
                    successorScore = result[1]

                    # Updates highest score so far
                    if (highestScore is None) or (successorScore > highestScore):
                        highestScore = successorScore
                        bestAction = action

            # Ghost's turn
            else:
                for action in legalActions:
                    nextGameState = gameState.generateSuccessor(agentIdx, action)

                    # Get the minimax score for the next agent
                    result = minimax(currentDepth, agentIdx + 1, nextGameState)
                    successorScore = result[1]

                    if (highestScore is None) or (successorScore < highestScore):
                        highestScore = successorScore
                        bestAction = action
            
            # Reached leaf node
            if highestScore is None:
                return None, self.evaluationFunction(gameState)
            
            return bestAction, highestScore 

        return minimax(INITIAL_DEPTH, PACMAN_IDX, gameState)[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphaBeta(self, currentDepth, agentIdx, gameState, alpha, beta):
        # Increase depth if all agents finished their turn
        if agentIdx >= gameState.getNumAgents():
            agentIdx = PACMAN_IDX
            currentDepth += 1

        # Return the value of evaluationFunction if max depth is reached
        if currentDepth == self.depth:
            return None, self.evaluationFunction(gameState)
        
        # Initialize best_score and best_action with None
        highestScore, bestAction = None, None

        # Pacman's turn
        if agentIdx == PACMAN_IDX:  
            for action in gameState.getLegalActions(agentIdx):
                nextGameState = gameState.generateSuccessor(agentIdx, action)
                
                # Get the score for the next agent
                result = self.alphaBeta(currentDepth, agentIdx + 1, nextGameState, alpha, beta)
                successorScore = result[1]

                # Updates highest score so far
                if (highestScore is None) or (successorScore > highestScore):
                    highestScore = successorScore
                    bestAction = action
                
                # Update the value of alpha
                alpha = max(alpha, successorScore)
                
                # Stop if condition is met
                if alpha > beta:
                    break
        
        # Ghost's turn
        else:  
            for action in gameState.getLegalActions(agentIdx):
                nextGameState = gameState.generateSuccessor(agentIdx, action)
                
                # Get the score for the next agent
                result = self.alphaBeta(currentDepth, agentIdx + 1, nextGameState, alpha, beta)
                successorScore = result[1]
                
                # Updates highest score so far
                if (highestScore is None) or (successorScore < highestScore):
                    highestScore = successorScore
                    bestAction = action
                
                beta = min(beta, successorScore)
                
                # Stop if condition is met
                if alpha > beta:
                    break

        # Reached leaf node
        if highestScore is None:
            return None, self.evaluationFunction(gameState)
        
        return bestAction, highestScore  # Return the best_action and highestScore

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphaBeta(INITIAL_DEPTH, PACMAN_IDX, gameState, -INFINITY, INFINITY)[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState, depth):
            legalActions = gameState.getLegalActions(PACMAN_IDX)

            if (len(legalActions) == 0) or gameState.isWin() or gameState.isLose() or (depth == self.depth):
                return self.evaluationFunction(gameState), None

            successRate = -INFINITY
            bestAction = None

            for action in legalActions:
                result = expValue(gameState.generateSuccessor(0, action), 1, depth)
                value = result[0]
                
                if successRate < value:
                    successRate, bestAction = value, action
                                                                                                   
            return successRate, bestAction

        def expValue(gameState, agentIdx, depth):
            legalActions = gameState.getLegalActions(agentIdx)

            if len(legalActions) == 0:
                return self.evaluationFunction(gameState), None

            successRate = 0

            for action in legalActions:
                if agentIdx == gameState.getNumAgents() - 1:
                    result = maxValue(gameState.generateSuccessor(agentIdx, action), depth + 1)
                else:
                    result = expValue(gameState.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                
                value = result[0]

                probability = value / len(legalActions)
                successRate += probability

            return successRate, None

        return maxValue(gameState, INITIAL_DEPTH)[1]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Essa função leva em consideração não somente a distância entre o pacman, os fantamas,
    os fantasmas assustados, a comida e as capsulas de poder, cada uma com uma ordem de importância 
    distinta:
        dist. capsulas -> qtd. comida restante -> dist. comida = dist. fantasma = dist. fantasma assustado
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    foodPositions = currentGameState.getFood()
    powerCapsulesPositions = currentGameState.getCapsules()

    if currentGameState.isLose():
        return -INFINITY
    if currentGameState.isWin():
        return INFINITY
    
    foodDistanceList = []
    for position in foodPositions.asList():
        foodDistanceList += [util.manhattanDistance(position, pacmanPosition)]
    
    minFoodDistance = min(foodDistanceList)
    
    ghostDistanceList = []
    scaredGhostsDistanceList = []
    for ghost in ghostStates:
        if ghost.scaredTimer == 0 :
            ghostDistanceList.append(util.manhattanDistance(pacmanPosition, ghost.getPosition()))
        elif ghost.scaredTimer > 0:
            scaredGhostsDistanceList.append(util.manhattanDistance(pacmanPosition, ghost.getPosition()))
    
    minGhostDistance = -1
    if len(ghostDistanceList) > 0:
        minGhostDistance = min(ghostDistanceList)
    
    minScaredGhostsDistanceList = -1
    if len(scaredGhostsDistanceList) > 0:
        minScaredGhostsDistanceList = min(scaredGhostsDistanceList)
    
    score = scoreEvaluationFunction(currentGameState)
    score -= 1.5 * minFoodDistance + 2 * (1.0/minGhostDistance) + 2 * minScaredGhostsDistanceList + 20 * len(powerCapsulesPositions) + 4 * len(foodPositions.asList())
    
    return score

# Abbreviation
better = betterEvaluationFunction
