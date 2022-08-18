from typing import Any

import numpy as np
import tensorflow as tf
from numpy.typing import ArrayLike
from tensorflow.keras.models import Model
from tqdm import trange

from pong.agent import DDPG, ReplayBuffer, SimpleAI
from pong.environment import (ACTION_SPACE, Batch, Environment, Field,
                              Transition)
from pong.renderer import Renderer


class DDPGModel(Model):
    def __init__(self, shape: ArrayLike) -> None:
        super().__init__()
        self.input_layer = tf.keras.layers.InputLayer(shape)
        self.layer_stack = [
            tf.keras.layers.Dense(units, activation=activation)
            for units, activation in zip(
                (32, 32, 32, 32, 1), ("tanh", "tanh", "tanh", "tanh", "tanh")
            )
        ]

    def call(self, inputs):
        x = tf.concat(inputs, axis=1)
        x = self.input_layer(x)
        for layer in self.layer_stack:
            x = layer(x)

        return x


def repack_batch(batch: Any, batch_size: int) -> Batch:
    # target = 0: player 1
    # target = 1: player 2

    # values of action enum items go from -1 to 1, hence +1 for the index
    action_batch = np.array(batch.action1, dtype="float32")

    # Gather states from batch
    state_batch = np.array(
        batch.state,
    ).reshape(batch_size, -1)

    new_state_batch = np.array(batch.new_state).reshape(batch_size, -1)

    reward_batch = batch.reward1
    reward_batch = np.array(reward_batch, dtype="float32")
    terminal_batch = np.array(batch.terminal, dtype="float32")

    return Batch(
        state_batch, action_batch, new_state_batch, reward_batch, terminal_batch
    )


def train_ddpg(episodes, batch_size, gamma, tau, num_freezes, mem_size):
    buffer = ReplayBuffer(mem_size)

    try:
        agent = DDPG.load("models/ddpg")
    except OSError:
        # Define actors per player
        actor = DDPGModel((None, 6))
        critique = DDPGModel((None, 7))

        agent = DDPG(actor, critique)

    env = Environment(Field())

    ai_agent = SimpleAI(env, 1)

    # Initialize training renderer
    renderer = Renderer(800, 400, env)

    # Training optimizer and loss

    loss = tf.keras.losses.MeanSquaredError()
    opt1 = tf.keras.optimizers.RMSprop(1e-5)
    opt2 = tf.keras.optimizers.RMSprop(1e-5)

    agent.compile(loss=loss, opt_actor=opt1, opt_critique=opt2)

    loss_values = []

    with trange(episodes, desc="episode") as t:
        cumulated_reward = 0

        for e in t:

            renderer.events()

            # One new step

            current_state = env.observe()

            action1 = agent.select_action(current_state[np.newaxis], noise=1.0)[0][0]
            action2 = ai_agent.select_action(current_state)

            nearest_action_index = np.argmin(
                [abs(action.value - action1) for action in ACTION_SPACE], axis=0
            )

            nearest_action = ACTION_SPACE[np.squeeze(nearest_action_index)]

            state, _, _, next_state, reward1, reward2, terminal = env.step(
                nearest_action, action2
            )

            cumulated_reward += reward1

            buffer.push(state, action1, action2, next_state, reward1, reward2, terminal)

            # Render current scene
            renderer.render()

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
                    agent.update_targets(tau)

                # Save networks
                if e % 5000 == 0 and e > 0:
                    agent.save("models/ddpg_2")

                if len(loss_values) > 500:
                    loss_values.clear()

    renderer.quit()


if __name__ == "__main__":
    EPISODES = 100000
    MEM_SIZE = 100000
    BATCH_SIZE = 512
    NUM_FREEZES = 1
    GAMMA = 0.999
    TAU = 0.999

    train_ddpg(EPISODES, BATCH_SIZE, GAMMA, TAU, NUM_FREEZES, MEM_SIZE)
