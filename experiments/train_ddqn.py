from operator import attrgetter
from typing import Any

import numpy as np
import tensorflow as tf
from numpy.typing import ArrayLike
from tensorflow.keras.models import Model
from tqdm import trange

from pong.agent import DDQN, ReplayBuffer, SimpleAI, epsilon_decay
from pong.environment import (ACTION_SPACE, Batch, Environment, Field,
                              Transition)
from pong.renderer import Renderer


class DQN(Model):
    def __init__(self, shape: ArrayLike) -> None:
        super().__init__()
        self.input_layer = tf.keras.layers.InputLayer(shape)
        self.conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", strides=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", strides=(2, 2))
        self.pool = tf.keras.layers.MaxPool2D(strides=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(3, activation="linear")

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x


def repack_batch(batch: Any, batch_size: int) -> Batch:
    # target = 0: player 1
    # target = 1: player 2

    # values of action enum items go from -1 to 1, hence +1 for the index
    action_indices = np.array(list(map(attrgetter("value"), batch.action1))) + 1

    # Gather states from batch
    state_batch = np.asarray(
        batch.state,
    )

    new_state_batch = np.asarray(batch.new_state)

    reward_batch = batch.reward1
    reward_batch = np.asarray(reward_batch, dtype="float32")
    terminal_batch = np.asarray(batch.terminal, dtype="float32")

    return Batch(
        state_batch, action_indices, new_state_batch, reward_batch, terminal_batch
    )


def train_ddqn(episodes, batch_size, gamma, tau, num_freezes, mem_size):
    buffer = ReplayBuffer(mem_size)

    agent = DDQN(DQN((None, 512, 256, 2)))

    env = Environment(Field())

    ai_agent = SimpleAI(env, 1)

    # Initialize training renderer
    renderer = Renderer(512, 256, env)

    # Training optimizer and loss

    loss = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.RMSprop(1e-5)

    agent.compile(loss=loss, optimizer=opt)

    loss_values = []

    with trange(episodes, desc="episode") as t:
        cumulated_reward = 0

        for e in t:
            renderer.events()

            current_state = renderer.observe()

            ai_state = env.observe()

            eps = epsilon_decay(e, int(0.8 * episodes), 0.2, 0.8, warm_up=e < 2000)

            action1 = agent.select_action(current_state[np.newaxis], eps=eps)
            action2 = ai_agent.select_action(ai_state)

            _, _, _, _, reward1, reward2, terminal = env.step(action1, action2)

            # Render current scene
            renderer.render()

            next_obs_state = renderer.observe()

            cumulated_reward += reward1

            buffer.push(current_state, action1, action2, next_obs_state, reward1, reward2, terminal)

            # Start training here
            if len(buffer) >= batch_size * 2:
                transitions = buffer.sample(batch_size)
                raw_batch = Transition(*zip(*transitions))

                batch = repack_batch(raw_batch, batch_size)

                # Update step for both agents
                loss_val = agent.optimize(batch, gamma)
                loss_values.append(loss_val)

                t.set_postfix(loss=np.mean(loss_values), total_reward=cumulated_reward)

                # Update target Q network and save models every num_freezes epochs
                if e % num_freezes == 0:
                    agent.update_target(tau)

                # Save networks
                if e % 5000 == 0 and e > 0:
                    agent.save("models/ddqn")

                if len(loss_values) > 500:
                    loss_values.clear()

    renderer.quit()


if __name__ == "__main__":
    EPISODES = 2000000
    MEM_SIZE = 100000
    BATCH_SIZE = 32
    NUM_FREEZES = 1
    GAMMA = 0.99
    TAU = 0.998

    train_ddqn(EPISODES, BATCH_SIZE, GAMMA, TAU, NUM_FREEZES, MEM_SIZE)
