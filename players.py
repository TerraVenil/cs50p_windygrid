import numpy as np

class Player():
    def __init__(self, name: str, state: tuple[int, int]) -> None:
        self.name = name[0].upper()
        self.state = state

    def update_state(self, state) -> None:
        self.state = state

    def get_action(self, state=None) -> int:
        raise NotImplementedError()

    def update_weights(self, action, reward, state_prime, action_prime) -> None:
        ...

class WindyAgent(Player):
    def __init__(self, name: str, state: tuple[int, int]) -> None:
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.9
        self.num_states = 7 * 10
        self.num_actions = 4
        self.weights = np.random.uniform(low=0, high=0.01, size=(self.num_states, self.num_actions))
        super().__init__(name, state)

    def get_action(self, state=None) -> int:
        state = self.__to_int(self.state if state is None else state)
        A = self.one_hot_encoding(state)
        q_A = np.matmul(A, self.weights)

        p = np.random.random()
        if p < self.epsilon:
            return np.random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(q_A[0])

    def update_weights(self, action, reward, state_prime, action_prime) -> None:
        state = self.__to_int(self.state)
        state_prime = self.__to_int(state_prime)

        q_value = self.get_q_value(state, action)

        expected_q_value_prime = 0
        action_max = np.argmax(self.weights[state_prime])
        for a in range(self.num_actions):
            if a == action_max:
                expected_q_value_prime += self.weights[state_prime][a] * self.epsilon
            else:
                expected_q_value_prime += self.weights[state_prime][a] / self.num_actions

        dw = self.alpha * (reward + self.gamma * expected_q_value_prime - q_value) * self.get_gradient(state, action)
        self.weights += dw

    def one_hot_encoding(self, state):
        return np.identity(self.num_states)[state:state + 1]

    def get_q_value(self, state, action):
        A = self.one_hot_encoding(state)
        return np.matmul(A, self.weights)[0][action]

    def get_gradient(self, state, action):
        feature = np.zeros((self.num_states, self.num_actions))
        feature[state][action] = 1
        return feature

    def __to_int(self, state) -> int:
        row, column = state
        return row * 10 + column

class Human(Player):
    def __init__(self, name: str, state: tuple[int, int]) -> None:
        super().__init__(name, state)

    def get_action(self, state=None) -> int:
        while True:
            try:
                action = int(input(""))
                if action in range(4):
                    return action
                raise ValueError("Not valid action. Please choose valid action")
            except ValueError as e:
                print(e)
                continue
