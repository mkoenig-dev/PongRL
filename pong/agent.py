import random

from tensorflow.keras.layers import Dense, InputLayer
import tensorflow as tf
from abc import ABC, abstractmethod
from .environment import Action, actions
import numpy as np
import pygame


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
