import random
from collections import deque
from functools import partial
from operator import attrgetter

import numpy as np
import tensorflow as tf
from tqdm import trange

from pong.agent import DDQN, DQN, UserAgent
from pong.environment import Batch, Environment, Field, Transition, state2vec
from pong.renderer import Renderer

class ReplayBuffer(object):
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.memory = deque([], mem_size)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def epsilon_decay(e, num_episodes, eps_start, eps_end, warm_up=False):
    if warm_up:
        return 0.0
    else:
        return (eps_end - eps_start) * ((e + 1) / num_episodes) + eps_start



def repack_batch(batch, batch_size, target=0):
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


def train_dqn(episodes, batch_size, gamma, tau, num_freezes, mem_size=10000):
    buffer = ReplayBuffer(mem_size)

    # Define actors per player
    policy1 = DQN()
    policy1.build((None, 6))

    policy2 = DQN()
    policy2.build((None, 6))

    target1 = DQN()
    target1.build((None, 6))
    target1.set_weights(policy1.get_weights())

    target2 = DQN()
    target2.build((None, 6))
    target2.set_weights(policy2.get_weights())

    agent1 = DDQN(policy1, target1)
    agent2 = DDQN(policy2, target2)

    env = Environment(Field())

    # Initialize training renderer
    renderer = Renderer(800, 400, env)

    # Training optimizer and loss

    loss1 = tf.keras.losses.Huber()
    loss2 = tf.keras.losses.Huber()
    optimizer1 = tf.keras.optimizers.RMSprop(0.00025)
    optimizer2 = tf.keras.optimizers.RMSprop(0.00025)

    agent1.dqn.compile(loss=loss1, optimizer=optimizer1)
    agent2.dqn.compile(loss=loss2, optimizer=optimizer2)

    # Random exploration
    eps_start = 0.1
    eps_end = 0.98

    loss_values = []

    with trange(episodes, desc="episode") as t:
        cumulated_reward = 0

        for e in t:

            renderer.events()

            # get epsilon greedy value
            eps = epsilon_decay(e, episodes, eps_start, eps_end, warm_up=e<1000)

            # One new step

            # for _ in trange(1, desc="explore"):
            current_state = env.observe()

            action1 = agent1.select_action(
                state2vec(current_state, target=0)[np.newaxis], eps=eps
            )

            action2 = agent2.select_action(
                state2vec(current_state, target=1)[np.newaxis], eps=eps
            )

            transition = env.act(action1, action2)

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
                loss_val1 = agent1.optimize(loss1, optimizer1, batch1, gamma)
                loss_val2 = agent2.optimize(loss2, optimizer2, batch2, gamma)

                loss_values.append(loss_val1)
                loss_values.append(loss_val2)

                t.set_postfix(loss=np.mean(loss_values), total_reward=cumulated_reward)


                # Update target Q network and save models every num_freezes epochs
                if e % num_freezes == 0:
                    agent1.update_target(tau)
                    agent2.update_target(tau)

                # Save networks
                if e % 5000 == 0:
                    agent1.save("models/ddqn1_new")
                    agent2.save("models/ddqn2_new")

                if len(loss_values) > 500:
                    loss_values.clear()


    renderer.quit()


if __name__ == "__main__":
    episodes = 100000
    batch_size = 32
    num_freezes = 1
    gamma = 0.99
    tau = 0.9

    train_dqn(episodes, batch_size, gamma, tau, num_freezes, mem_size=50000)
