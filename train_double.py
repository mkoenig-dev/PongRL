from functools import partial
from operator import attrgetter
from typing import Any

import numpy as np
import tensorflow as tf
from tqdm import trange

from pong.agent import DDQN, QModel, ReplayBuffer, epsilon_decay
from pong.environment import Batch, Environment, Field, Transition, state2vec
from pong.renderer import Renderer


def repack_batch(batch: Any, batch_size: int, target: int = 0) -> Batch:
    # target = 0: player 1
    # target = 1: player 2

    # values of action enum items go from -1 to 1, hence +1 for the index
    action_batch = batch.action1 if target == 0 else batch.action2
    action_indices = np.array(list(map(attrgetter("value"), action_batch))) + 1

    # Gather states from batch
    state_batch = np.array(
        list(map(partial(state2vec, target=target), batch.state))
    ).reshape(batch_size, -1)

    new_state_batch = np.array(
        list(map(partial(state2vec, target=target), batch.new_state))
    ).reshape(batch_size, -1)

    reward_batch = batch.reward1 if target == 0 else batch.reward2
    reward_batch = np.array(reward_batch, dtype="float32")
    terminal_batch = np.array(batch.terminal, dtype="float32")

    return Batch(
        state_batch, action_indices, new_state_batch, reward_batch, terminal_batch
    )


def train_ddqn(episodes, batch_size, gamma, tau, num_freezes, mem_size):
    buffer = ReplayBuffer(mem_size)

    # Define actors per player
    policy1 = QModel((None, 6))
    policy2 = QModel((None, 6))

    agent1 = DDQN(policy1)
    agent2 = DDQN(policy2)

    env = Environment(Field())

    # Initialize training renderer
    renderer = Renderer(800, 400, env)

    # Training optimizer and loss

    loss1 = tf.keras.losses.Huber()
    loss2 = tf.keras.losses.Huber()
    optimizer1 = tf.keras.optimizers.Adam(0.000025)
    optimizer2 = tf.keras.optimizers.Adam(0.000025)

    agent1.compile(loss=loss1, optimizer=optimizer1)
    agent2.compile(loss=loss2, optimizer=optimizer2)

    # Random exploration
    eps_start = 0.1
    eps_end = 0.98

    loss_values = []

    with trange(episodes, desc="episode") as t:
        cumulated_reward = 0

        for e in t:

            renderer.events()

            # get epsilon greedy value
            eps = epsilon_decay(e, episodes, eps_start, eps_end, warm_up=e < 1000)

            # One new step

            # for _ in trange(1, desc="explore"):
            current_state = env.observe()

            action1 = agent1.select_action(
                state2vec(current_state, target=0)[np.newaxis], eps=eps
            )

            action2 = agent2.select_action(
                state2vec(current_state, target=1)[np.newaxis], eps=eps
            )

            transition = env.step(action1, action2)

            cumulated_reward += transition.reward1 + transition.reward2

            buffer.push(*transition)

            # Render current scene
            renderer.render()

            # Start training here
            if len(buffer) >= batch_size:
                transitions = buffer.sample(batch_size)
                raw_batch = Transition(*zip(*transitions))

                batch1 = repack_batch(raw_batch, batch_size, target=0)
                batch2 = repack_batch(raw_batch, batch_size, target=1)

                # Update step for both agents
                loss_val1 = agent1.optimize(batch1, gamma)
                loss_val2 = agent2.optimize(batch2, gamma)

                loss_values.append(loss_val1)
                loss_values.append(loss_val2)

                t.set_postfix(loss=np.mean(loss_values), total_reward=cumulated_reward)

                # Update target Q network and save models every num_freezes epochs
                if e % num_freezes == 0:
                    agent1.update_target(tau)
                    agent2.update_target(tau)

                # Save networks
                if e % 5000 == 0:
                    agent1.save("models/ddqn1_new_2")
                    agent2.save("models/ddqn2_new_2")

                if len(loss_values) > 500:
                    loss_values.clear()

    renderer.quit()


if __name__ == "__main__":
    EPISODES = 1000000
    MEM_SIZE = 100000
    BATCH_SIZE = 512
    NUM_FREEZES = 1
    GAMMA = 0.98
    TAU = 0.99

    train_ddqn(EPISODES, BATCH_SIZE, GAMMA, TAU, NUM_FREEZES, MEM_SIZE)
