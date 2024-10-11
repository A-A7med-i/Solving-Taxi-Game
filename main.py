from algorithm.QLearning import QLearningAgent
from algorithm.Sarsa import SarsaAgent
import numpy as np
import gym


def render_episode(env, q_values):
    frames = []
    state = env.reset()
    frames.append(env.render(mode="rgb_array"))
    done = False

    while not done:
        action = np.argmax(q_values[state])
        state, reward, done, info = env.step(action)
        frames.append(env.render(mode="rgb_array"))

    return frames


if __name__ == "__main__":
    env = gym.make("Taxi-v3")

    q1_learner = QLearningAgent(env)
    q1 = q1_learner.train()
    q2_learner = SarsaAgent(env)
    q2 = q2_learner.train()

    render_episode(env, q1)
    render_episode(env, q2)
