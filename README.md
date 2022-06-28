# Pacman AI Agent Project
This project was implemented from [Berkeley CS 188](https://inst.eecs.berkeley.edu/~cs188/fa20/project1/#introduction). In this project, I build blind search, multi-agent search, and reinforcement learning algorithms to teach the Pacman agent to find paths through his maze world, both to reach a particular location and to collect food efficiently. 

## Pacman Blind Search
For this project section, I build blind search algorithms including Breadth First Search, Depth First Search, and A* search. 

## Multi-Agent Search
In this project section, I build algorithms for Pacman to find his path with the minimax, alpha beta pruning, and expectimax to collect its food, and escape the ghosts while blinking. 

## Reinforcement Learning
In this project section, I implement value iteration and Q-learning for Pacman to find his way through the maze. 

# Usage
1. Clone the repository
2. Open the terminal and run the following:
```
cd dir
(command)
```
We can use the command below to run the project 

# List of commands 
```
// For classic pacman game (you play)
python pacman.py 
// Easy 
python pacman.py -p ReflexAgent -l testClassic
// Reflex Agent with one ghost / two ghost 
python pacman.py --frameTime 0 -p ReflexAgent -k 1
python pacman.py --frameTime 0 -p ReflexAgent -k 2
// Depth first search
python pacman.py -l mediumMaze -p SearchAgent
//Breadth first search
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
// A* Search
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
// Minimax
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
// Alpha beta Agent on small map
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
// Expectimax Agent
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
// Q Learning
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid
// Approximate Q Learning
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid
```

# Video overview
<img src='http://i.imgur.com/IEpdZWC.gif' title='pacman' width='' alt='Video Walkthrough' />

The above shows the pacman behavior on reflex agent with two ghost.
