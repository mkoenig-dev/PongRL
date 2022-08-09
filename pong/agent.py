import random

from tensorflow.keras.layers import Dense
import tensorflow as tf
from abc import ABC, abstractmethod
from .environment import Action, actions
import numpy as np
import pygame
from pathlib import Path


class DQN(tf.keras.Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.x1 = Dense(32, activation="relu")
        self.x2 = Dense(64, activation="relu")
        self.x2 = Dense(32, activation="relu")
        self.x3 = Dense(3, activation="relu")

    def call(self, x):
        x = self.x1(x)
        x = self.x2(x)
        out = self.x3(x)

        return out


class Agent(ABC):
    @abstractmethod
    def select_action(self, state) -> Action:
        pass


class DQNAgent(Agent):
    def __init__(self, actor):
        self.actor = actor

    def select_action(self, states, eps=None) -> Action:
        if eps is not None and random.random() > eps:
            return random.choice(actions)

        q_approx = self.actor(states)
        i = np.argmax(q_approx)

        return actions[i]


class DDQN(Agent):
    def __init__(self, dqn, target_dqn) -> None:
        self.dqn = dqn
        self.target_dqn = target_dqn

    @tf.function
    def optimize(self, loss, optimizer, state_batch, action_indices, new_state_batch, reward_batch, gamma):
        with tf.GradientTape() as tape:
            # Calculate target Q value for chosen actions
            q_values = tf.gather(
                self.dqn(state_batch),
                action_indices,
                axis=1
            )

            # Calculate policy q value based on max Q
            best_next_actions = tf.argmax(self.dqn(new_state_batch), axis=1)
            next_target_q_vals = self.target_dqn(new_state_batch)
            inter_q_values = tf.gather(
                next_target_q_vals,
                best_next_actions,
                axis=1
            )
            target_q = reward_batch + gamma * inter_q_values

            # Calculate Huber loss
            loss_value = loss(q_values, target_q)

        # Get gradients w.r.t. weights
        grads = tape.gradient(loss_value, self.dqn.trainable_variables)

        # Update policy network
        optimizer.apply_gradients(zip(grads, self.dqn.trainable_variables))

    def select_action(self, states, eps=None) -> Action:
        if eps is not None and random.random() > eps:
            return random.choice(actions)

        q_approx = self.dqn(states)
        i = tf.squeeze(tf.argmax(q_approx, axis=1))

        return actions[i]

    def update_target(self):
        self.target_dqn.set_weights(self.dqn.get_weights())

    def save(self, path):
        path = Path(path)

        if not path.exists():
            path.mkdir()

        try:
            self.dqn.save(path / "dqn")
            self.target_dqn.save(path / "target")
        except Exception:
            raise RuntimeError(f"Could not save model to {path}")

    def load(path):
        path = Path(path)
        dqn_path = path / "dqn"
        target_path = path / "target"
        
        if path.exists() and dqn_path.exists() and target_path.exists():
            try:
                dqn = tf.keras.models.load_model(dqn_path)
                target = tf.keras.models.load_model(target_path)

                return DDQN(dqn, target)
            except Exception:
                raise RuntimeError(f"Unable to load DDQN from {path}")


class UserAgent(Agent):
    def select_action(self, state, event=None) -> Action:
        action = Action.STILL
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                action = Action.DOWN
            elif event.key == pygame.K_DOWN:
                action = Action.UP
        elif event.type == pygame.KEYUP:
            action = Action.STILL

        return action
