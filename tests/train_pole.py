import gym
from collections import deque, namedtuple
import random
from tqdm import trange
import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np

from pong.agent import DQNModel


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))
Batch = namedtuple(
    "Batch", ("states", "actions_indices", "new_states", "rewards", "terminal")
)


class QModel(tf.keras.Model):
    def __init__(self):
        super(QModel, self).__init__()
        self.x1 = Dense(128, activation="relu")
        self.x2 = Dense(64, activation="relu")
        self.x3 = Dense(2)

    def call(self, x):
        x = self.x1(x)
        x = self.x2(x)
        out = self.x3(x)

        return out


class ReplayBuffer(object):
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.memory = deque([], mem_size)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        raw_batch = Transition(*zip(*transitions))
        actions = np.array(raw_batch.action)
        states = np.array(raw_batch.state)
        rewards = np.array(raw_batch.reward)
        next_states = np.array(raw_batch.next_state)
        terminals = np.array(raw_batch.done)

        return Batch(states, actions, next_states, rewards, terminals)

    def __len__(self):
        return len(self.memory)


def epsilon_decay(e, num_episodes, eps_start, eps_end, warm_up=False):
    if warm_up:
        return 0.0
    else:
        return (eps_end - eps_start) * ((e + 1) / num_episodes) + eps_start


def train_dqn(episodes, batch_size, gamma, tau, mem_size):
    env = gym.make("CartPole-v1", render_mode="human", new_step_api=True)
    model = QModel()

    target = QModel()
    target.set_weights(model.get_weights())

    loss = tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss=loss, optimizer=optimizer)

    dqn = DQNModel(model, target, env.action_space)

    buffer = ReplayBuffer(mem_size)

    with trange(episodes, desc="episode") as t:
        for e in t:
            terminated = False
            state = env.reset()

            eps = epsilon_decay(e, round(0.8 * episodes), 0.9, 0.02, e < 0.1 * episodes)

            losses = []

            while not terminated:
                action = dqn.select_action(state[np.newaxis, :], eps)
                observation, reward, terminated, truncated, info = env.step(action.numpy())
                
                buffer.push(state, action, reward, observation, truncated)
                state = observation

                # Training step
                if len(buffer) > batch_size:
                    batch = buffer.sample(batch_size)
                    loss_val = dqn.optimize(loss, optimizer, batch, gamma)
                    losses.append(loss_val.numpy())
                    
                    dqn.update_target(tau)

                t.set_postfix(loss=np.mean(losses))





if __name__ == "__main__":
    train_dqn(
        episodes=10000,
        batch_size=16,
        gamma=0.999,
        tau=0.8,
        mem_size=100000
    )
