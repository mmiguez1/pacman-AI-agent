# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


# function returns list of actions to lead agent from start to goal
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # Fringe implemented as LIFO Stack
    fringe = util.Stack()
    visited = []

    # Create start node from start state & action
    start_state = problem.getStartState()
    startNode = (start_state, [])

    # Add start node to fringe
    fringe.push(startNode)

    while not fringe.isEmpty():

        # Choose top node in fringe
        getNode = fringe.pop()

        # Check if the current state is goal state
        if problem.isGoalState(getNode[0]):
            return getNode[1]

        # Check if node has been visited
        if getNode[0] not in visited:
            visited.append(getNode[0])

            # Expand node to see children
            for child_node in problem.getSuccessors(getNode[0]):

                # Get child node's state & the sequence of actions to get to child node
                getChild = (child_node[0], getNode[1] + [child_node[1]])

                # If child node has not been visited yet, add to fringe
                if getChild[0] not in visited:
                    fringe.push(getChild)


# function returns list of actions to lead agent from start to goal
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    # Implement fringe as FIFO Queue
    fringe = util.Queue()
    visited = []

    # Create start node from start state & action
    start_state = problem.getStartState()
    startNode = (start_state, [])

    # Add start node to fringe
    fringe.push(startNode)

    while not fringe.isEmpty():

        # Choose top node in fringe
        getNode = fringe.pop()

        # Check if the state is goal state
        if problem.isGoalState(getNode[0]):
            return getNode[1]

        # Check if node has been visited
        if getNode[0] not in visited:
            visited.append(getNode[0])

            # Expand node to see children
            for child_node in problem.getSuccessors(getNode[0]):

                # Get child node's state & the sequence of actions to get to child node
                getChild = (child_node[0], getNode[1] + [child_node[1]])

                # If child node has not been visited yet, add to fringe
                if getChild[0] not in visited:
                    fringe.push(getChild)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    
    # Implement fringe as priority queue
    fringe = util.PriorityQueue()
    visited = []

    # Create start node from start state, action, & step cost
    start_state = problem.getStartState()
    startNode = (start_state, [], 0)

    # Add start node & its cost to fringe
    fringe.push(startNode, startNode[2])

    while not fringe.isEmpty():

        # Choose top node in fringe
        getNode = fringe.pop()

        # Check if the state is goal state
        if problem.isGoalState(getNode[0]):
            return getNode[1]

        # Check if node has been visited
        if getNode[0] not in visited:
            visited.append(getNode[0])

            # Expand node to see children
            for child_node in problem.getSuccessors(getNode[0]):

                # Get child node's state, the sequence of actions to get to child node, & its total cost
                getChild = (child_node[0], getNode[1] + [child_node[1]], child_node[2] + getNode[2])

                # If child node has not been visited yet, add node & its cost to fringe
                if getChild[0] not in visited:
                    fringe.push(getChild, getChild[2])


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # Implement fringe as priority queue
    fringe = util.PriorityQueue()
    visited = []

    # Create start node from start state, action, & step cost + heuristic
    start_state = problem.getStartState()
    startNode = (start_state, [], 0)

    # Add start node & its cost to fringe
    fringe.push(startNode, startNode[2])

    while not fringe.isEmpty():

        # Choose top node in fringe
        getNode = fringe.pop()

        # Check if the state is goal state
        if problem.isGoalState(getNode[0]):
            return getNode[1]

        # Check if node has been visited or not
        if getNode[0] not in visited:
            visited.append(getNode[0])

            # Expand node to see children
            for child_node in problem.getSuccessors(getNode[0]):

                # Get child node's state, the sequence of actions to get to child node, & its total cost f(n) = g(n) + h(n)
                getChild = (child_node[0], getNode[1] + [child_node[1]], problem.getCostOfActions(getNode[1] + [child_node[1]])+ heuristic(child_node[0], problem))

                # If child node has not been visited yet, add node & its cost + heuristic to fringe
                if child_node[0] not in visited:
                    fringe.push(getChild, getChild[2])

                

                
           



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
