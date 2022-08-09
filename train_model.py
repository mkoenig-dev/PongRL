import random
from collections import deque
from pong.environment import Environment, Transition, Field, state2vec
from pong.agent import DQN, DQNAgent
from tqdm import trange
import numpy as np
import tensorflow as tf
from functools import partial
from operator import attrgetter


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


def train_step(policy, target, loss, optimizer, state_batch, action_indices, reward_batch, gamma):
    with tf.GradientTape() as tape:
        # Calculate target Q value for chosen actions
        policy_q_value = tf.gather(
            policy(state_batch),
            action_indices,
            axis=1
        )

        # Calculate policy q value based on max Q
        next_action_values = reward_batch + gamma * tf.reduce_max(target(state_batch), axis=1)

        # Calculate Huber loss
        loss_value = loss(policy_q_value, next_action_values)

    # Get gradients w.r.t. weights
    grads = tape.gradient(loss_value, policy.trainable_variables)

    # Update policy network
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))


def train_dqn(episodes, batch_size, gamma, num_freezes, mem_size=10000):
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

    agent1 = DQNAgent(policy1)
    agent2 = DQNAgent(policy2)

    env = Environment(Field())

    # Training optimizer and loss

    loss1 = tf.keras.losses.Huber()
    loss2 = tf.keras.losses.Huber()
    optimizer1 = tf.keras.optimizers.Adam()
    optimizer2 = tf.keras.optimizers.Adam()

    # Random exploration
    eps_start = 0.3
    eps_end = 0.99

    for e in trange(episodes, desc="episode"):

        # get epsilon greedy value
        eps = (eps_end - eps_start) * ((e+1) / episodes) + eps_start

        # Fill buffer with transition pairs
        for _ in trange(mem_size, desc="explore"):
            current_state = env.observe()

            action1 = agent1.select_action(
                np.expand_dims(state2vec(current_state, target="agent"), axis=0),
                eps=eps
            )
            action2 = agent2.select_action(
                np.expand_dims(state2vec(current_state, target="opponent"), axis=0),
                eps=eps
            )

            transition = env.act(action1, action2)
            buffer.push(*transition)

        # Loop over batches
        for _ in trange(num_freezes, desc="steps"):

            # Start training here
            transitions = buffer.sample(batch_size)
            batch = Transition(*zip(*transitions))

            # Gather action indices from transitions
            action_indices1 = np.array(list(map(attrgetter("value"), batch.action1))) + 1
            action_indices2 = np.array(list(map(attrgetter("value"), batch.action2))) + 1

            # Gather states from transitions
            state_batch1 = np.array(
                list(map(state2vec, batch.state))
            ).reshape(batch_size, -1)

            state_batch2 = np.array(
                list(map(partial(state2vec, target="opponent"), batch.state))
            ).reshape(batch_size, -1)

            # Gather rewards from transitions
            reward_batch1 = np.array(batch.reward1, dtype="float32")
            reward_batch2 = np.array(batch.reward2, dtype="float32")

            # Update step for both agents
            train_step(policy1, target1, loss1, optimizer1, state_batch1, action_indices1, reward_batch1, gamma)
            train_step(policy2, target2, loss2, optimizer2, state_batch2, action_indices2, reward_batch2, gamma)

        # Update target Q network and save models

        target1.set_weights(policy1.get_weights())
        agent1.actor = policy1

        target2.set_weights(policy2.get_weights())
        agent2.actor = policy2

        target1.save("models/target1")
        target2.save("models/target2")


if __name__ == "__main__":
    episodes = 1000
    batch_size = 1000
    num_freezes = 100
    gamma = 0.999

    train_dqn(episodes, batch_size, gamma, num_freezes)
