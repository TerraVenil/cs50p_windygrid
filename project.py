import numpy as np
import pickle
from pyfiglet import Figlet
from consolemenu import *
from consolemenu.items import *
from players import Player, Human, WindyAgent
from windy import WindyGrid
from progress.bar import ChargingBar
from tabulate import tabulate

def load_agent(name: str) -> WindyAgent:
    with open(f'{name}.pickle', 'rb') as f:
        return pickle.load(f)

def save_agent(agent: Player) -> None:
    with open(f'{agent.name}.pickle', 'wb') as f:
        pickle.dump(agent, f)

def start_game():
    env = WindyGrid()
    agent = load_agent('W')
    human = Human(name='H', state=(5, 0))
    players: list[Player] = [human, agent]

    start_states = np.random.choice(6, 2, replace=False)
    for idx, p in enumerate(players):
        p.update_state((start_states[idx], 0))

    def action_to_str(action):
        return list(env.actions.keys())[list(env.actions.values()).index(action)]

    done = False
    while not done:
        for p in players:
            print(f"Player '{p.name}' is at position {p.state}. Choose an action 0 - up, 1 - right, 2 - down, 3 - left")
            env.render(players)

            state, action = p.state, p.get_action()
            print(f"Action {action_to_str(action)} has been choosen by player '{p.name}'")

            # With probability 20% we migh choose random action instead of direction corresponding to player action
            action = np.random.choice([np.random.randint(0, 3), action], 1, p=[0.2, 0.8])[0]
            print(f"Action {action_to_str(action)} has been choosen by environment")

            state_prime, reward, done = env.step(state, action)

            if isinstance(p, WindyAgent):
                action_prime = p.get_action(state_prime)
                p.update_weights(action, reward, state_prime, action_prime)

            p.update_state(state_prime)

            print(f"Environment moved player '{p.name}' to {state_prime}")

            if done:
                if isinstance(p, WindyAgent):
                    save_agent(p)
                env.render(players)
                input(f"Player '{p.name}' won!")
                break

def train():
    env = WindyGrid()
    start_state = (3, 0)
    agent = WindyAgent(name='T', state=start_state)
    players = [agent]
    verosity = False

    number_of_episodes = 1_000
    bar = ChargingBar('Training', max=number_of_episodes)
    for n in range(number_of_episodes):
        done = False
        player = env.reset(players)
        player.update_state((np.random.randint(0, 6), 0))

        if verosity:
            env.render(players)
            print(f"Player '{player.name}' is at position {player.state}")
        state, action = player.state, player.get_action()
        while not done:
            if verosity:
                print(f"Action {list(env.actions.keys())[list(env.actions.values()).index(action)]} was choosen")

            # if n >= 1:
            #     print(f"Episode number {n}")
            #     env.render(players)
            #     input("Next?")

            state_prime, reward, done = env.step(state, action)

            if verosity:
                print(f"Environment moved player '{player.name}' to {state_prime}")

            action_prime = player.get_action(state_prime)

            player.update_weights(action, reward, state_prime, action_prime)

            player.update_state(state_prime)

            if verosity:
                env.render(players)

            state = state_prime
            action = action_prime

        bar.next()
        if n == number_of_episodes - 1:
            bar.finish()
            save_agent(agent)

def main():
    figlet = Figlet()
    figlet.getFonts()
    figlet.setFont(font="slant")

    menu = ConsoleMenu("Welcome to ", figlet.renderText("Windy World"))

    start_new_game = FunctionItem("Start a new game", function=start_game, should_exit=False)
    train_agent = FunctionItem("Train an agent", function=train, should_exit=False)

    prologue_text = '''The windy grid world task is another navigation task, where the agent has to find its way from start to goal. The grid has a height of 7 and a width of 10 squares. There is a wind blowing in the 'up' direction in the middle part of the grid, with a strength of 1 or 2 depending on the column. Again, the agent can choose between four movement actions: up, down, left and right, each resulting in a reward of -1. The result of an action is a movement of 1 square in the corresponding direction plus an additional movement in the 'up' direction, corresponding with the wind strength. For example, when the agent is in the square right of the goal and takes a 'left' action, it ends up in the square just above the goal.'''
    epilogue_text = '''Stochasticity was added to the environment by moving the agent with a probability of 20% in a random direction instead of the direction corresponding to the action.'''

    help_menu = ConsoleMenu(title="Game rules", subtitle=tabulate([["Source",'Sutton and A. Barto. Reinforcement learning: An introduction. Second Edition. Example 6.5.']], tablefmt="simple", maxcolwidths=[None, 64]),\
        prologue_text=prologue_text,\
        epilogue_text=epilogue_text)
    game_rules = SubmenuItem("Game rules", help_menu, menu=menu)

    menu.append_item(start_new_game)
    menu.append_item(train_agent)
    menu.append_item(game_rules)

    menu.show()
    menu.join()

if __name__ == "__main__":
    main()