from .RLAlgorithm import RLAlgorithm
import numpy as np


class QLearningAgent(RLAlgorithm):
    """
    Represents a Q-learning agent for reinforcement learning tasks.

    Args:
        env: The environment to interact with.
        num_episodes: The total number of training episodes.
        exploration_rate: The probability of exploring random actions.
        learning_rate: The learning rate for updating Q-values.
        discount_factor: The discount factor for future rewards.
    """

    def __init__(
        self,
        env,
        num_episodes: int = 1000,
        exploration_rate: float = 0.2,
        learning_rate: float = 0.2,
        discount_factor: float = 0.9,
    ):
        super().__init__(
            env, num_episodes, exploration_rate, learning_rate, discount_factor
        )

    def update_q_values(
        self,
        q_values: np.ndarray,
        state: int,
        action: int,
        new_state: int,
        reward: float,
    ) -> np.ndarray:
        """
        Updates the Q-values based on the current state, action, new state, and reward.

        Args:
            q_values: The current Q-values table.
            state: The current state.
            action: The chosen action.
            new_state: The new state after taking the action.
            reward: The reward received for the action.

        Returns:
            The updated Q-values table.
        """
        q_values[state, action] = q_values[state, action] + self.learning_rate * (
            (reward + self.discount_factor * np.max(q_values[new_state, :]))
            - q_values[state, action]
        )

        return q_values

    def train(self) -> np.ndarray:
        """
        Trains the Q-learning agent.

        Returns:
            The final Q-values table.
        """
        q_values = np.zeros([self.env.observation_space.n, self.env.action_space.n])

        for episode in range(self.num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:

                action = self.choose_action(state, q_values)

                new_state, reward, done, info = self.env.step(action)

                q_values = self.update_q_values(
                    q_values, state, action, new_state, reward
                )

                total_reward += reward
                state = new_state

            self.reward.append(total_reward)

        return q_values
