from players import WindyAgent
from project import save_agent, load_agent, train
import os

def test_save_agent():
    save_agent(WindyAgent('T', (3, 0)))
    assert os.path.exists(f"{os.path.join(os.getcwd(), 'T.pickle')}")

def test_load_agent():
    agent = WindyAgent('T', (3, 0))
    save_agent(agent)
    assert isinstance(load_agent('T'), WindyAgent)

def test_train():
    train()
    assert os.path.exists(f"{os.path.join(os.getcwd(), 'W.pickle')}")