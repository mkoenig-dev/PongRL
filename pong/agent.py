import random
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pygame
import tensorflow as tf
from tensorflow.keras.layers import Dense

from .environment import Action, Ball, Field, Player, actions


class DQN(tf.keras.Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.x1 = Dense(256, activation="relu")
        self.x2 = Dense(128, activation="relu")
        self.out = Dense(3, activation="linear")

    def call(self, x):
        x1 = self.x1(x)
        x2 = self.x2(x1)
        out = self.out(x2)

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

    def optimize(self, loss, optimizer, batch, gamma):
        # Unpack batch
        states = batch.states
        action_indices = batch.actions_indices
        new_states = batch.new_states
        rewards = batch.rewards
        terminal = batch.terminal

        next_target_q_vals = self.target_dqn(new_states)
        inter_q_values = tf.reduce_max(next_target_q_vals, axis=1)
        target_q = rewards + gamma * inter_q_values * (1.0 - terminal)

        loss_value = self.step(loss, optimizer, states, action_indices, target_q)

        return loss_value

    @tf.function
    def step(self, loss, optimizer, states, action_indices, target_q):
        with tf.GradientTape() as tape:
            # Calculate target Q value for chosen actions
            # q_values = tf.gather(self.dqn(states), action_indices, axis=1)
            q_values = tf.math.reduce_sum(
                self.dqn(states) * tf.one_hot(action_indices, 3), axis=1
            )

            # Calculate policy q value based on max Q
            # best_next_actions = tf.argmax(self.dqn(new_states), axis=1)

            # inter_q_values = tf.gather(next_target_q_vals, best_next_actions, axis=1)

            # Calculate Huber loss
            loss_value = loss(q_values, target_q)

        # Get gradients w.r.t. weights
        grads = tape.gradient(loss_value, self.dqn.trainable_variables)
        # clipped_grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]

        # Update policy network
        # optimizer.apply_gradients(zip(clipped_grads, self.dqn.trainable_variables))
        optimizer.apply_gradients(zip(grads, self.dqn.trainable_variables))

        return loss_value

    def select_action(self, states, eps=None) -> Action:
        if eps is not None and random.random() > eps:
            return random.choice(actions)

        q_approx = self.dqn(states)
        i = tf.squeeze(tf.argmax(q_approx, axis=1))

        return actions[i]

    def update_target(self, tau=0.99):
        self.target_dqn.set_weights(
            [
                (1.0 - tau) * w_dqn + tau * w_target
                for w_dqn, w_target in zip(
                    self.dqn.get_weights(), self.target_dqn.get_weights()
                )
            ]
        )

    def save(self, path: str):
        path = Path(path)

        if not path.exists():
            path.mkdir()

        try:
            self.dqn.save(path / "dqn")
            self.target_dqn.save(path / "target")
        except Exception:
            raise RuntimeError(f"Could not save model to {path}")

    def load(path: str):
        load_path = Path(path)
        dqn_path = load_path / "dqn"
        target_path = load_path / "target"

        if load_path.exists() and dqn_path.exists() and target_path.exists():
            try:
                dqn = tf.keras.models.load_model(dqn_path)
                target = tf.keras.models.load_model(target_path)

                return DDQN(dqn, target)
            except Exception:
                raise RuntimeError(f"Unable to load DDQN from {path}")
        else:
            raise OSError("One of the paths for loading does not exist.")


class SimpleAI(Agent):
    def __init__(self, field: Field, ball: Ball, player: Player) -> None:
        self.field = field
        self.ball = ball
        self.player = player
        self.target = int(player.pos_x > self.field.origin[0] + 0.5 * self.field.width)
        self.agent = "agent" if self.target == 0 else "opponent"
        self.last_action = Action.STILL
        self.real_target_pos = np.zeros(2)
        self.target_pos = np.zeros(2)

    def predict_intersection(self, state: np.ndarray, depth=0):

        ball_pos = state[2:4]
        ball_dir = state[4:6]

        # top and bottom wall y_pos
        vrange = np.array(
            [
                self.field.origin[1] + self.ball.radius,
                self.field.origin[1] + self.field.height - self.ball.radius,
            ]
        )

        # get left and right goal line
        hrange = np.array(
            [
                self.field.origin[0] + self.ball.radius,
                self.field.origin[0] + self.field.width - self.ball.radius,
            ]
        )

        # estimate number of steps to hit the wall
        ysteps = -(vrange - ball_pos[1]) / (ball_dir[1] * self.field.speed)
        ysteps = np.ceil(ysteps.max())

        xsteps = (hrange - ball_pos[0]) / (ball_dir[0] * self.field.speed)
        xsteps = np.ceil(xsteps[self.target])

        if xsteps > 0 and depth < 5:
            if ysteps > xsteps:
                return ball_pos + self.field.speed * ball_dir * xsteps
            else:

                ball_pos += self.field.speed * ball_dir * ysteps
                ball_dir[1] = -ball_dir[1]

                # Recursive call on new position
                next_state = np.array([state[0], state[1], *ball_pos, *ball_dir])
                return self.predict_intersection(next_state, depth=depth + 1)
        else:
            return np.array([hrange[self.target], 0.5 * vrange.sum()])

    def select_action(self, state: np.ndarray) -> Action:
        target_pos = self.predict_intersection(state)
        target_diff = target_pos - self.real_target_pos

        if target_diff.dot(target_diff) > 2.0:
            # set new swiffled target
            self.real_target_pos = target_pos
            self.target_pos = target_pos.copy()
            self.target_pos[1] -= random.random() * self.player.height

        player_pos = state[self.target]
        ydiff = self.target_pos[1] - player_pos

        if random.random() > 0.9:
            return self.last_action

        if abs(ydiff) < 1.0:
            self.last_action = Action.STILL
        elif ydiff < 0:
            self.last_action = Action.DOWN
        elif ydiff > 0:
            self.last_action = Action.UP
        else:
            self.last_action = Action.STILL

        return self.last_action


class UserAgent(Agent):
    def __init__(self):
        self.action = Action.STILL

    def select_action(self, state, event=None) -> Action:
        if event is not None:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.action = Action.DOWN
                elif event.key == pygame.K_DOWN:
                    self.action = Action.UP
            elif event.type == pygame.KEYUP:
                self.action = Action.STILL

        return self.action


class DQNModel(Agent):
    def __init__(self, dqn, target, action_space) -> None:
        self.dqn = dqn
        self.target_dqn = target
        self.action_space = action_space

    def optimize(self, loss, optimizer, batch, gamma):
        # Unpack batch
        states = batch.states
        action_indices = batch.actions_indices
        new_states = batch.new_states
        rewards = batch.rewards
        terminal = batch.terminal

        # Calculate policy q value based on max Q
        # best_next_actions = tf.argmax(self.dqn(new_states), axis=1)

        next_target_q_vals = self.target_dqn(new_states)

        # inter_q_values = tf.gather(next_target_q_vals, best_next_actions, axis=1)
        inter_q_values = tf.reduce_max(next_target_q_vals, axis=1)
        target_q = rewards + (gamma * inter_q_values * (1.0 - terminal))

        loss_value = self.step(loss, optimizer, states, action_indices, target_q)

        return loss_value

    @tf.function
    def step(self, loss, optimizer, states, action_indices, target_q):
        with tf.GradientTape() as tape:
            # Calculate target Q value for chosen actions
            q_values = tf.gather(self.dqn(states), action_indices, axis=1)

            # Calculate Huber loss
            loss_value = loss(q_values, target_q)

        # Get gradients w.r.t. weights
        grads = tape.gradient(loss_value, self.dqn.trainable_variables)
        clipped_grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]

        # Update policy network
        optimizer.apply_gradients(zip(clipped_grads, self.dqn.trainable_variables))

    def select_action(self, states, eps=None) -> Action:
        if eps is not None and random.random() < eps:
            return self.action_space.sample()

        q_approx = self.dqn(states)
        i = tf.squeeze(tf.argmax(q_approx, axis=1))

        return i

    def update_target(self, tau=0.99):
        self.target_dqn.set_weights(
            [
                (1.0 - tau) * w_dqn + tau * w_target
                for w_dqn, w_target in zip(
                    self.dqn.get_weights(), self.target_dqn.get_weights()
                )
            ]
        )

    def save(self, path: str):
        path = Path(path)

        if not path.exists():
            path.mkdir()

        try:
            self.dqn.save(path / "dqn")
            self.target_dqn.save(path / "target")
        except Exception:
            raise RuntimeError(f"Could not save model to {path}")

    def load(path: str):
        load_path = Path(path)
        dqn_path = load_path / "dqn"
        target_path = load_path / "target"

        if load_path.exists() and dqn_path.exists() and target_path.exists():
            try:
                dqn = tf.keras.models.load_model(dqn_path)
                target = tf.keras.models.load_model(target_path)

                return DDQN(dqn, target)
            except Exception:
                raise RuntimeError(f"Unable to load DDQN from {path}")
        else:
            raise OSError("One of the paths for loading does not exist.")
