from abc import ABC
from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class RLAlgorithm(ABC):
    """
    Abstract base class for reinforcement learning algorithms.

    This class provides a basic framework for implementing various
    reinforcement learning algorithms. Subclasses must implement the
    specific learning logic in the `learn` method.

    Args:
        env: The GYM environment to interact with.
        num_episodes: The number of training episodes.
        exploration_rate: The probability of exploring random actions.
        learning_rate: The learning rate for updating Q-values or other parameters.
        discount_factor: The discount factor for future rewards.
    """

    def __init__(
        self,
        env,
        num_episodes: int = 10000,
        exploration_rate: float = 0.2,
        learning_rate: float = 0.2,
        discount_factor: float = 0.9,
    ):
        self.env = env
        self.num_episodes = num_episodes
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.reward = []

    @abstractmethod
    def train(self) -> np.ndarray:
        """
        Performs the training of the reinforcement learning agent.

        This method should implement the specific learning logic
        of the chosen RL algorithm, such as updating Q-values or
        policy parameters.

        Returns:
            The final learned parameters (e.g., Q-values) or None if not applicable.
        """
        pass

    def choose_action(self, state: int, q_values: np.ndarray) -> int:
        """
        Chooses an action based on an epsilon-greedy policy.

        This method implements an epsilon-greedy policy where the
        agent explores with probability `exploration_probability` or
        exploits (chooses the best action) otherwise.

        Args:
            state: The current state.
            q_values: The current Q-value table (if applicable).

        Returns:
            The chosen action.
        """

        if np.random.uniform(0, 1) < self.exploration_rate:
            return self.env.action_space.sample()
        else:
            return np.argmax(q_values[state, :])

    def plot_performance(self):
        """
        Visualizes the learning performance over training episodes.

        This method plots the total reward earned in each episode.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.reward)
        plt.title("Learning Performance")
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.show()
