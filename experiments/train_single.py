from copy import copy
from operator import attrgetter
from typing import Any

import numpy as np
import tensorflow as tf
from tqdm import trange

from pong.agent import DDQN, QModel, ReplayBuffer, SimpleAI, epsilon_decay
from pong.environment import Batch, Environment, Field, Transition
from pong.renderer import Renderer


def repack_batch(batch: Any, batch_size: int) -> Batch:
    # target = 0: player 1
    # target = 1: player 2

    # values of action enum items go from -1 to 1, hence +1 for the index
    action_batch = batch.action1
    action_indices = np.array(list(map(attrgetter("value"), action_batch))) + 1

    # Gather states from batch
    state_batch = np.array(
        batch.state,
    ).reshape(batch_size, -1)

    new_state_batch = np.array(batch.new_state).reshape(batch_size, -1)

    reward_batch = batch.reward1
    reward_batch = np.array(reward_batch, dtype="float32")
    terminal_batch = np.array(batch.terminal, dtype="float32")

    return Batch(
        state_batch, action_indices, new_state_batch, reward_batch, terminal_batch
    )


def train_ddqn(episodes, batch_size, gamma, tau, num_freezes, mem_size):
    buffer = ReplayBuffer(mem_size)

    try:
        agent = DDQN.load("models/ddqn")
        agent.target = copy(agent.policy)
    except OSError:
        # Define actors per player
        policy = QModel((None, 6))

        agent = DDQN(policy)

    env = Environment(Field())

    ai_agent = SimpleAI(env, 1)

    # Initialize training renderer
    renderer = Renderer(800, 400, env)

    # Training optimizer and loss

    loss = tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(5e-3)

    agent.compile(loss=loss, optimizer=optimizer)

    # Random exploration
    eps_start = 0.9
    eps_end = 0.9

    loss_values = []

    with trange(episodes, desc="episode") as t:
        cumulated_reward = 0

        for e in t:
            renderer.events()

            # get epsilon greedy value
            eps = epsilon_decay(e, episodes, eps_start, eps_end, warm_up=False)

            # One new step

            # for _ in trange(1, desc="explore"):
            current_state = env.observe()

            action1 = agent.select_action(current_state[np.newaxis], eps=eps)

            action2 = ai_agent.select_action(current_state)

            state, action1, action2, next_state, reward1, reward2, terminal = env.step(
                action1, action2
            )

            cumulated_reward += reward1

            buffer.push(state, action1, action2, next_state, reward1, reward2, terminal)

            # Render current scene
            renderer.render()

            # Start training here
            if len(buffer) >= batch_size:
                transitions = buffer.sample(batch_size)
                raw_batch = Transition(*zip(*transitions))

                batch = repack_batch(raw_batch, batch_size)

                # Update step for both agents
                loss_val = agent.optimize(batch, gamma)
                loss_values.append(loss_val)

                t.set_postfix(
                    loss=np.mean(loss_values), total_reward=cumulated_reward, eps=eps
                )

                # Update target Q network and save models every num_freezes epochs
                if e % num_freezes == 0:
                    agent.update_target(tau)

                # Save networks
                if e % 5000 == 0 and e > 0:
                    agent.save("models/ddqn_corr")

                if len(loss_values) > 500:
                    loss_values.clear()

    renderer.quit()


if __name__ == "__main__":
    EPISODES = 100000
    MEM_SIZE = 100000
    BATCH_SIZE = 512
    NUM_FREEZES = 1
    GAMMA = 0.98
    TAU = 0.99

    train_ddqn(EPISODES, BATCH_SIZE, GAMMA, TAU, NUM_FREEZES, MEM_SIZE)
