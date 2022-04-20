# multiAgents.py
# --------------
# This codebase is adapted from UC Berkeley AI. Please see the following information about the license.
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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #Find the distance to the nearest food in the game state
        #Our for loop will determine the closest dot(food) to our pacman so it can absorb it and add it to our score
        #Use the manhattan distance to calculate this
        foodLocated = newPos
        currScore = 0
        
        #This will add a point to our score if a food is detected in the current position
        if (currentGameState.getFood()[newPos[0]][newPos[1]]):
            currScore +=1
        
        for dot in newFood:
            closestFood = min(newFood, key=lambda x: manhattanDistance(x, foodLocated))
            currScore = currScore + 1 / manhattanDistance(closestFood, foodLocated)
            #Replace the closest food as a marked down food that has been located.
            foodLocated = closestFood
            newFood.remove(closestFood)
        #Add all the scores together to get the total
        currScore += successorGameState.getScore()
            
        #This will determing if the ghost runs into the pacman. If it does it will evaluate it as a loss and will subtract 100000 from the score to initialize the Loss.
        if (min([manhattanDistance(ghost.getPosition(), newPos) for ghost in newGhostStates]) == 1):
            return -999999
        #Return the final score
        return currScore
        
        util.raiseNotDefined()

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
    #WIll be used for the minimum agents
    def getMinNum(self, gameState,depth, agentIndex):
        #This will check if there are any more real moves that can be conducted
        lengthOfGame = len(gameState.getLegalActions(agentIndex))
        if (lengthOfGame == 0):
            return self.evaluationFunction(gameState)

        #Will determine the last ghost
        if (gameState.getNumAgents() - 1 > agentIndex):
            return min([self.getMinNum(gameState.generateSuccessor(agentIndex, action),depth, agentIndex + 1) for action in gameState.getLegalActions(agentIndex)])
        else:
            return min([self.getMaxNum(gameState.generateSuccessor(agentIndex, action), depth + 1) for action in gameState.getLegalActions(agentIndex)])
            
    #Will be used for the maximum agents
    def getMaxNum(self, gameState,depth):
    #This will check if there are any more legal actions that can be conducted
        lengthOfGame = len(gameState.getLegalActions(0))
        if((lengthOfGame == 0) or (depth == self.depth)):
            return self.evaluationFunction(gameState)
        return max([self.getMinNum(gameState.generateSuccessor(0, action), 1, depth) for action in gameState.getLegalActions(0)])
    

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

        negInf = float("-inf")
        maximumVal = negInf
        maximumAction = None
        legalActions = gameState.getLegalActions(0)
        
        #This will use our moves in the game to get the maximum value returned by the previous successor moves of pacman
        for move in legalActions:
            direction = self.getMinNum(gameState.generateSuccessor(0, move),0,1)
            #Compare the values to determine the maximum of all of the successors
            if(direction > maximumVal):
                maximumVal = direction
                maximumAction = move
        #Will return the maximum
        return maximumAction
        util.raiseNotDefined()
        
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    
    #Get the minimum number of points required to make a particular move
    def getMinNum(self, gameState,depth, negInf, inf, agentIndex):
        lengthAgentIndex = len(gameState.getLegalActions(agentIndex))
        legalActions = gameState.getLegalActions(agentIndex)
        movePoints = float('inf')
        
        if (lengthAgentIndex == 0):
            return self.evaluationFunction(gameState)
        
        for move in legalActions:
            if (gameState.getNumAgents() - 1 > agentIndex):
                movePoints = min(movePoints, self.getMinNum(gameState.generateSuccessor(agentIndex,move), depth, negInf, inf, agentIndex + 1))
            else:
                movePoints = min(movePoints, self.getMaxNum(gameState.generateSuccessor(agentIndex, move), depth + 1, negInf, inf))
            #This will continue to replace the minimum number of points needed for each move to reuse in the comparisons
            if (negInf > movePoints):
                return movePoints
            inf = min(inf, movePoints)
        return movePoints
        
    def getMaxNum(self, gameState, depth, negInf, inf):
        #Stores the legal action length and amounts like the other max and min methods
        lengthLegalActions = len(gameState.getLegalActions(0))
        legalActions = gameState.getLegalActions(0)
        movePoints = float('-inf')
        
        #If there are no legal actions that can be done or it is the final depth it can go
        if (lengthLegalActions == 0 or depth == self.depth):
            return self.evaluationFunction(gameState)

        #Calculate the total points for the moves
        for move in legalActions:
            movePoints = max(movePoints, self.getMinNum(gameState.generateSuccessor(0, move), depth, negInf, inf,1))
            if (inf < movePoints):
                return movePoints
            negInf = max(negInf, movePoints)

        return movePoints
    
    
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #the inf variables will initially be infinities but will be reinitialized when new variables come that are new mins or maxs
        negInf = float('-inf')
        posInf = float('inf')
        totalMoves = None
        legalActions = gameState.getLegalActions(0)
        movePoints = negInf
        
        #For every legal move that is made, we will assign each move a certain amount of points
        for decision in legalActions:
            movePoints = self.getMinNum(gameState.generateSuccessor(0, decision),0, negInf, posInf,1)
            #IF the value of our points is greater than negative infinity, it will be reassigned
            if (movePoints > negInf):
            #Then reassign negative infinity so we can update our comparison values to determine if our amount of points that was made for hte move was more effective or not
                negInf = movePoints
                totalMoves = decision
        return totalMoves

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    #Same as the Minimax Function class getMax
    #Will be used for the maximum agents
    def getMaxNum(self, gameState,depth):
    #This will check if there are any more legal actions that can be conducted
        lengthOfGame = len(gameState.getLegalActions(0))
        totalMoves = gameState.getLegalActions(0)
        
        if((lengthOfGame == 0) or (depth == self.depth)):
            return self.evaluationFunction(gameState)
        return max([self.getMinNum(gameState.generateSuccessor(0, action), 1, depth) for action in totalMoves])
            
    #Again we can use it just like the minimax class
    #WIll be used for the minimum agents
    def getMinNum(self, gameState,depth, agentIndex):
        #This will check if there are any more real moves that can be conducted
        lengthOfGame = len(gameState.getLegalActions(agentIndex))
        totalMoves = gameState.getLegalActions(agentIndex)
        
        if (lengthOfGame == 0):
            return self.evaluationFunction(gameState)
            #Will determine where the last ghost is
        if (gameState.getNumAgents() - 1 > agentIndex):
            return sum([self.getMinNum(gameState.generateSuccessor(agentIndex, move),depth, agentIndex +1) for move in totalMoves]) / float(lengthOfGame)
        else:
            return sum([self.getMaxNum(gameState.generateSuccessor(agentIndex, move), depth + 1) for move in totalMoves]) / float(lengthOfGame)
            
        
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
    
        #These will count as all the possible legal directions that pacman will be able to move
        legalActions = gameState.getLegalActions(0)
        negInf = float('-inf')
        movePoints  = None
        costOfMove = 0
    
        #Will go through all the successors to get the max of them
        for move in legalActions:
            costOfMove = self.getMinNum(gameState.generateSuccessor(0, move), 0, 1)
            #This will always compare these two values so we can get the max from the pair involved
            if (negInf < costOfMove):
                negInf = costOfMove
                movePoints = move
        return movePoints

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    """
    With this method, we decided to make it so that our pacman only attempts at gobling the food when it knows for certain that the pacman is of a certain distance. We made our code ensure that the distance of the ghost was of a certain distance compared to pacman. In order to do this, we used the manhattan distance in order to calculate the minimum distances and to calculate the distance from the pacman to the ghosts. We also compared the number of capsules so we can use them to get rid of the pacmans if they were in valid distance.
    
    """
    "*** YOUR CODE HERE ***"
    #These will be used to find our food dots, organize our lists, and see the position of our pacman compared to our ghosts and agents
    newFood = currentGameState.getFood()
    foodList = newFood.asList()
    newPos = currentGameState.getPacmanPosition()
    #This will check  how many of the big dots there are that allow the ghosts to be eaten. Use the built in functions of the .getCapsules to check for them
    bigDots = currentGameState.getCapsules()
    countOfBigDots = len(bigDots)
    
    #This will calculate the range from the ghosts to pacman and whether they are right beside him or not
    ghostDistance = 1
    ghostRange = 0
    ghostPosition = currentGameState.getGhostPositions()
    for ghostLocation in ghostPosition:
        distance = util.manhattanDistance(newPos, ghostLocation)
        ghostDistance += distance
        if distance <= 1:
            ghostRange += 1
    
    #This will check for the nearest location of the dots in the food
    nearestDot = -1
    for eachDot in foodList:
        dist = util.manhattanDistance(newPos, eachDot)
        if  nearestDot == -1 or nearestDot >= dist:
            nearestDot = dist

    #This will take all of that above to get the scores and ranges
    return (1 / float(nearestDot)) + currentGameState.getScore() - countOfBigDots - (1 / float(ghostDistance)) - ghostRange

# Abbreviation
better = betterEvaluationFunction
