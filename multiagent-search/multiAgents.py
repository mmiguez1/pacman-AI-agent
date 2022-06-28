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

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        ghosts = []
        ghostDist = 0
       
        # Find food closest to current position using manhattan distance
        foods = [manhattanDistance(newPos, food) for food in newFood.asList()]
       
        # Add closest food distance to current score (Going for closer food = Higher Score)
        if len(foods) > 0:
            closeDist = min(foods)
            score += 5/closeDist+1
        else:
            closeDist = 0

        # Find distance of ghosts to current position using manhattan distance
        for ghost in newGhostStates:
            ghostDist = manhattanDistance(ghost.getPosition(), newPos)
            ghosts.append(ghostDist)

        # If ghost is at a distance of 1 or less, subtract 5 points from score
            if ghostDist < 2:
                score += -5

        # Subtract closest ghost distance from current score (Closer ghost = Lower Score)
        closeGhost = min(ghosts)
        if closeGhost == 0:
            return -1000000
        else:
            score -= 5/closeGhost+1
                        
        return score
        

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

    def getAction(self, gameState):
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

        def minimax_helper(self, gameState, agentIndex, depth):

            getLegalActions = gameState.getLegalActions(agentIndex)
            PACMAN = 0
            nextAgentIndex = agentIndex + 1
            max_val = -10000
            min_val = 10000
            max_action = None
            
            # If game state is winning or losing state or the last level of tree,
            # return evaluation function value
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Check if agent is Pacman (Max player)
            if agentIndex == PACMAN:

                # Find maximum value of max node's successors
                for action in getLegalActions:
                    val = minimax_helper(self, gameState.generateSuccessor(agentIndex, action), 1, depth)
                    if val > max_val:
                        max_val = val
                        max_action = action

                # If depth of tree is reached, return best action for Pacman
                if depth != 0:
                    return max_val
                else:
                    return max_action

            # Check if agent is ghost (Min player)
            else:
                # Check if all agents took turn, and if yes, next agent is Pacman 
                # and adjust the depth for the next ply
                if agentIndex + 1 == gameState.getNumAgents(): 
                    nextAgentIndex = PACMAN
                    depth += 1

                # Find minimum value of Ghost node's successors
                for action in getLegalActions:
                    val = minimax_helper(self, gameState.generateSuccessor(agentIndex, action), nextAgentIndex, depth)
                    if val < min_val:
                        min_val = val
                       
                return min_val
        
        return minimax_helper(self, gameState, self.index, 0)
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        
        def max_val(gameState, alpha, beta, current_depth):
            # If game state is winning or losing state or the last level of tree,
            # return evaluation function value
            if gameState.isWin() or gameState.isLose() or current_depth == self.depth:
                return self.evaluationFunction(gameState)
            
            # Take max value from min nodes
            v = float('-inf')
            for action in gameState.getLegalActions(0):
                v = max(v, min_val(gameState.generateSuccessor(0, action), alpha, beta, current_depth, 1))
                # If value is > beta, prune following nodes
                if v > beta:
                    return v
                alpha = max(alpha, v)
            
            return v

        def min_val(gameState, alpha, beta, current_depth, agentIndex):
            # If game state is winning or losing state or the last level of tree,
            # return evaluation function value
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # Take min value from max nodes
            # Let each ghost play before returning value
            v = float('inf')
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex < gameState.getNumAgents() - 1: #excluding last ghost agent
                    v = min(v, min_val(gameState.generateSuccessor(agentIndex, action), alpha, beta, current_depth, agentIndex + 1))

                # Last ghost agent plays
                else:
                    v = min(v, max_val(gameState.generateSuccessor(agentIndex, action), alpha, beta, current_depth+1))
                if v < alpha:
                    return v
                beta = min(beta, v)

            return v

        # Let Pacman play first & return the best action for him to take
        val = float('-inf')
        alpha = float('-inf')
        max_action = None
        for action in gameState.getLegalActions(0):
            val = min_val(gameState.generateSuccessor(0, action), alpha, float('inf'), 0, 1)
            if val > alpha:
                alpha = val
                max_action = action

        return max_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax_helper(self, gameState, agentIndex, depth):

            getLegalActions = gameState.getLegalActions(agentIndex)
            PACMAN = 0
            nextAgentIndex = agentIndex + 1
            max_val = -10000
            min_val = 10000
            max_action = None
            
            # Probability of each node
            if len(getLegalActions) > 1:
                probability = 1 / len(getLegalActions)
            else:
                probability = 1
            
            # If game state is winning or losing state or the last level of tree,
            # return evaluation function value
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Check if agent is Pacman (Max player)
            if agentIndex == PACMAN:

                # Find maximum value of max node's successors
                for action in getLegalActions:
                    val = expectimax_helper(self, gameState.generateSuccessor(agentIndex, action), 1, depth)
                    if val > max_val:
                        max_val = val
                        max_action = action

                # If entire depth of tree is reached, return best action for Pacman
                if depth != 0:
                    return max_val
                else:
                    return max_action

            # Check if agent is ghost (Min player)
            else:
                # Check if all agents took turn, and if yes, next agent is Pacman 
                # and adjust the depth for the next ply
                if agentIndex + 1 == gameState.getNumAgents(): 
                    nextAgentIndex = PACMAN
                    depth += 1

                # Find expectimax values for node's successors
                values = [probability * expectimax_helper(self, gameState.generateSuccessor(agentIndex, action), \
                    nextAgentIndex, depth) for action in getLegalActions]
                
                # Return sum of expectimax values for node
                return sum(values)
        
        return expectimax_helper(self, gameState, self.index, 0)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    score = currentGameState.getScore()
    ghosts = []
    ghostDist = 0
       
    # Find food closest to current position using manhattan distance
    foods = [manhattanDistance(newPos, food) for food in newFood.asList()]
       
    # Add closest food distance to current score (Closer food = Higher Score)
    if len(foods) > 0:
        closeDist = min(foods)
        score += 5/closeDist+1
    else:
        closeDist = 0

    # Find distance of ghosts to current position using manhattan distance
    for ghost in newGhostStates:
        ghostDist = manhattanDistance(ghost.getPosition(), newPos)
        ghosts.append(ghostDist)

    # If ghost is at a distance of 1 or less, subtract 5 points from score
        if ghostDist < 2:
            score += -5
    
    # Subtract closest ghost distance from current score (Closer ghost = Lower Score)
    closeGhost = min(ghosts)
    if closeGhost == 0:
        return -1000000
    else:
        score -= 5/closeGhost+1

    #If ghost is scared, add 50 points to score. If not, subtract 5 points from score
    for scaredTime in newScaredTimes:
        if scaredTime <= 0:
            score += -5
        else:
            score += 50
                    
    return score

# Abbreviation
better = betterEvaluationFunction

