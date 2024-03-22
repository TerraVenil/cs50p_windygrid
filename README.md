# Wingy Grid
#### Video Demo: [CS50P. Windy Grid](https://youtu.be/Be6ml03lThg)
#### Description: The project is a 7x10 grid turn-based game with an element of stochasticity, where one player is an AI agent and another player is human.

#### The project implements the following tasks:

- find an optimal path in grid environment with an element of stochasticity
- implementation of Expected SARSA algorithm inspired by [Theoretical and Empirical Analysis of Expected Sarsa](https://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/vanseijenadprl09.pdf)
- instead of a tabular approach for an Q-value function for educational purposes I decided to choose function approximation by neural network with no hidden layers
- exposed ability to train own agent by adjusting parameters such as learning rate, number of episodes, ratio for exploration and exploitation
- modification of [Windy Gridworld environment](https://github.com/ibrahim-elshar/gym-windy-gridworlds) to allow human to play with trained agent

```
📦project
 ┣ 📜README.md
 ┣ 📜W.pickle
 ┣ 📜main_menu.png
 ┣ 📜players.py
 ┣ 📜project.py
 ┣ 📜requirements.txt
 ┣ 📜test_project.py
 ┗ 📜windy.py
```

`project.py` file contains main menu with different options and you can navigate between them
1. Start a new game
2. Train an agent. Keep in mind if you want to train a new agent again from scratch it might take some time but you will be able to track this progress.
3. Game rules
4. Exit

Moreover, the functions to store and load the pre-trained agent from `W.pickle` file to be able to play with a human player on a good enough level. Every time you start a game the trained agent will be loaded from this file.

Basic class Player and inherited from it Human and Agent are presented in `players.py` and implement logic of picking actions, updating state on the grid and weights based on neural network approximation of Q-value calculated by Expected SARSA algorithm.

The `windy.py` introduces a gridworld environment and exposes a render function to display the grid itself and player positions.