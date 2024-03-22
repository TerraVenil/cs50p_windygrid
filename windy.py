import numpy as np
import sys
from players import Player

class WindyGrid():
    def __init__(self, grid_height=7, grid_width=10,\
                wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],\
                goal_state = (3, 7),\
                reward = -1) -> None:
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.wind = wind
        self.goal_state = goal_state
        self.reward = reward
        self.action_space = 4
        self.observation_space = self.grid_height * self.grid_width
        self.actions = { 'U':0,   #up
                         'R':1,   #right
                         'D':2,   #down
                         'L':3 }  #left

        self.action_destination = np.empty((self.grid_height,self.grid_width), dtype=dict)
        for i in range(0, self.grid_height):
            for j in range(0, self.grid_width):
                destination = dict()
                destination[self.actions['U']] = (max(i - 1 - self.wind[j], 0), j)
                destination[self.actions['D']] = (max(min(i + 1 - self.wind[j], \
                                                    self.grid_height - 1), 0), j)
                destination[self.actions['L']] = (max(i - self.wind[j], 0),\
                                                       max(j - 1, 0))
                destination[self.actions['R']] = (max(i - self.wind[j], 0),\
                                                   min(j + 1, self.grid_width - 1))
                self.action_destination[i,j]=destination
        self.nA = len(self.actions)

    def step(self, state, action):
        observation = self.action_destination[state][action]
        if observation == self.goal_state:
            return observation, -1, True
        return observation, -1, False

    def reset(self, players: list[Player]) -> Player:
        observation = np.random.choice(players)
        return observation

    def render(self, players: list[Player]) -> None:
        outfile = sys.stdout
        nS = self.grid_height * self.grid_width
        shape = (self.grid_height, self. grid_width)

        def is_player(position):
            for p in players:
                if p.state == position:
                    return True
            return False

        def get_player_name(position):
            for p in players:
                if p.state == position:
                    return p.name
            return " "

        outboard = ""
        for y in range(-1, self.grid_height + 1):
            outline = ""
            for x in range(-1, self.grid_width + 1):
                position = (y, x)
                if is_player(position):
                    output = f" {get_player_name(position)} "
                elif position == self.goal_state:
                    output = " G "
                elif x in {-1, self.grid_width } or y in {-1, self.grid_height}:
                    output = " # "
                else:
                    output = " - "

                if position[1] == shape[1]:
                    output += '\n'
                outline += output
            outboard += outline
        outboard += '\n'
        outfile.write(outboard)