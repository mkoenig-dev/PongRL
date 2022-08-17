from pickletools import optimize
import random
from abc import ABC, abstractmethod
from copy import copy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.random as npr
import pygame
import tensorflow as tf
from numpy.typing import NDArray
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from .environment import Action, Ball, Batch, Field, Player, actions


class DQN(tf.keras.Model):
    def __init__(self):
        """Simple multilayer perceptron implementation for policy/target network"""
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
    def select_action(self, state: NDArray) -> Action:
        """Select an environment action based on the passed state.

        Args:
            state (NDArray): the passed state

        Returns:
            Action: the action selected by the agent
        """
        pass


class EpsilonGreedyAgent(Agent):
    def __init__(self, actor: Model):
        """Epsilon-greedy agent for keras models / action-value function approximations.

        Args:
            actor (Model): the q-value approimation model.
        """
        self.actor = actor

    def select_action(self, states: NDArray, eps: Optional[float] = None) -> Action:
        """Select action based on epsilon-greedy policy.

        Args:
            states (NDArray): the passed state
            eps (float, optional): Probability for random action. Defaults to None.

        Returns:
            Action: the action selected by the agent
        """
        if eps is not None and random.random() > eps:
            return random.choice(actions)

        q_approx = self.actor(states)
        i = np.argmax(q_approx)

        return actions[i]


class DDQN(Agent):
    def __init__(self, model: Model) -> None:
        """Dual Deep Q-Network implementation.

        Args:
            model (Model): the initial policy/target model
        """
        self.policy = model
        self.target = copy(model)
        self.optimizer = None
        self.loss = None

    def compile(self, loss: Any, optimizer: Any) -> None:
        self.optimizer = optimizer
        self.loss = loss

    def optimize(self, batch: Batch, gamma: float) -> float:
        """Trains policy network in one step on a minibatch.

        Args:
            loss (Any): loss function for target q update
            optimizer (Any): optimizer
            batch (Batch): minibatch with states, actions, next_states, rewards and
              terminal booleans
            gamma (float): discount factor

        Returns:
            float: loss value of the minibatch prediction.
        """
        if self.optimizer is None or self.loss is None:
            raise ValueError("Please compile the agent before training.")

        # Unpack batch
        states = batch.states
        actions = batch.actions_indices
        next_states = batch.new_states
        rewards = batch.rewards
        terminal = batch.terminal

        next_target_q_vals = self.target(next_states)
        inter_q_values = tf.reduce_max(next_target_q_vals, axis=1)
        target_q = rewards + gamma * inter_q_values * (1.0 - terminal)

        loss_value = self.step(self.loss, self.optimizer, states, actions, target_q)

        return loss_value

    @tf.function
    def step(self, loss, optimizer, states, actions, target_q) -> float:
        """Optimization step for policy update. Outsourced to its own function
        to make use of tensorflow autograph features.

        Args:
            loss (Any): loss function for target q update
            optimizer (Any): optimizer
            states (Any): minibatch states
            actions (Any): minibatch actions
            target_q (Any): target Q-value

        Returns:
            float: loss value of the minibatch prediction.
        """
        with tf.GradientTape() as tape:
            # Calculate target Q value for chosen actions
            # q_values = tf.gather(self.dqn(states), action_indices, axis=1)
            q_values = tf.math.reduce_sum(
                self.policy(states) * tf.one_hot(actions, 3), axis=1
            )

            # Calculate policy q value based on max Q
            # best_next_actions = tf.argmax(self.dqn(new_states), axis=1)

            # inter_q_values = tf.gather(next_target_q_vals, best_next_actions, axis=1)

            # Calculate Huber loss
            loss_value = loss(q_values, target_q)

        # Get gradients w.r.t. weights
        grads = tape.gradient(loss_value, self.policy.trainable_variables)
        clipped_grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]

        # Update policy network
        optimizer.apply_gradients(zip(clipped_grads, self.policy.trainable_variables))

        return loss_value

    def select_action(self, states: NDArray, eps: Optional[float] = None) -> Action:
        """Select action based on epsilon-greedy policy.

        Args:
            states (NDArray): the passed state
            eps (float, optional): Probability for random action. Defaults to None.

        Returns:
            Action: the action selected by the agent
        """
        if eps is not None and random.random() > eps:
            return random.choice(actions)

        q_approx = self.policy(states)
        i = tf.squeeze(tf.argmax(q_approx, axis=1))

        return actions[i]

    def update_target(self, tau: float = 0.0):
        """Updates target network softly by linear interpolation weighted by tau.
        Default behavior is a complete weight transfer from policy to target (tau=0.0).

        Args:
            tau (float, optional): Percentage of remaining target weights.
              Defaults to 0.0.
        """
        if 0.0 <= tau <= 1.0:
            self.target.set_weights(
                [
                    (1.0 - tau) * w_policy + tau * w_target
                    for w_policy, w_target in zip(
                        self.policy.get_weights(), self.target.get_weights()
                    )
                ]
            )
        else:
            raise ValueError(
                f"Interpolation parameter tau exceeds ranges [0, 1], {tau}"
            )

    def save(self, path: str):
        """Saves ddqn model to file using tf-model saving.
        Will create two folders named "dqn" and "target" in path.

        Args:
            path (str): path to the saved model files

        Raises:
            RuntimeError: If models could not be saved
        """
        path = Path(path)

        if not path.exists():
            path.mkdir()

        try:
            self.policy.save(path / "dqn")
            self.target.save(path / "target")
        except Exception:
            raise RuntimeError(f"Could not save model to {path}")

    def load(path: str) -> "DDQN":
        """Loads saved DDQN from specified model folder.
        Path must have two sub folders named "dqn" and "target" containing the model
        files.

        Args:
            path (str): saved model directory

        Raises:
            RuntimeError: If DDQN could not be loaded
            OSError: If one of the sub model files do not exist

        Returns:
            DDQN: new instance created from the loaded models
        """
        load_path = Path(path)
        dqn_path = load_path / "dqn"
        target_path = load_path / "target"

        if load_path.exists() and dqn_path.exists() and target_path.exists():
            try:
                policy = tf.keras.models.load_model(dqn_path)
                target = tf.keras.models.load_model(target_path)

                ddqn = DDQN(policy)
                ddqn.target = target

                return ddqn
            except Exception:
                raise RuntimeError(f"Unable to load DDQN from {path}")
        else:
            raise OSError("One of the paths for loading does not exist.")


class SimpleAI(Agent):
    def __init__(self, field: Field, ball: Ball, player: Player) -> None:
        """Simple Pong AI that predicts intersections of the ball on the goal plane.

        Args:
            field (Field): pong field
            ball (Ball): pong ball
            player (Player): the player to control
        """
        self.field = field
        self.ball = ball
        self.player = player
        self.target = int(player.pos_x > self.field.origin[0] + 0.5 * self.field.width)
        self.agent = "agent" if self.target == 0 else "opponent"
        self.last_action = Action.STILL
        self.real_target_pos = np.zeros(2)
        self.target_pos = np.zeros(2)

    def predict_intersection(self, state: NDArray, depth: int = 0) -> NDArray:
        """Predicts the intersection point of the ball trajectory with the AI's
        goal plane from the passed state.

        Args:
            state (NDArray): the state for prediction
            depth (int, optional): Number of bounces to track. Defaults to 0.

        Returns:
            NDArray: the intersection point if the calculation resolves in the number of
              recursive bounce calls (depth). Otherwise the target position is the
              vertical position of the ball.
        """

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

    def transform_state(self, state: NDArray) -> NDArray:
        """Transforms state back from normalized ranges to screen ranges.

        Args:
            state (NDArray): the normalized state

        Returns:
            NDArray: the transformed state
        """
        return np.array(
            [
                0.5 * self.field.height * (state[0] + 1.0),
                0.5 * self.field.height * (state[1] + 1.0),
                0.5 * self.field.width * (state[2] + 1.0),
                0.5 * self.field.height * (state[3] + 1.0),
                state[4] * self.ball.ball_speed,
                state[5] * self.ball.ball_speed,
            ],
            dtype="float32",
        )

    def select_action(self, state: NDArray) -> Action:

        state_t = self.transform_state(state)

        # Calculate player's target position

        target_pos = self.predict_intersection(state_t)
        target_diff = target_pos - self.real_target_pos

        # Change current target on huge variance to last target

        if target_diff.dot(target_diff) > 2.0:
            # set new swiffled target
            self.real_target_pos = target_pos
            self.target_pos = target_pos.copy()
            self.target_pos[1] -= random.random() * self.player.height

        player_pos = state_t[self.target]
        ydiff = self.target_pos[1] - player_pos

        # Make AI do mistakes by chance
        # Retakes the last action in 10% of the time

        if random.random() > 0.9:
            return self.last_action

        # Otherwise determine action based on distance to target position
        # If target position is too close, stay still

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
        """Agent based on user input"""
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


class OUNoise:
    def __init__(
        self, dim: int, mu: float = 0.0, sigma: float = 0.2, theta: float = 0.15,
    ) -> None:
        """
        Implements a discrete Ornstein-Uhlenbeck process to generate noise for the
        DDPG action policy.

        Args:
            dim (int): The dimension of the random variable.
            mu (float, optional): The average noise. Defaults to 0.0.
            theta (float, optional): Stiffness parameter. Defaults to 0.15.
            sigma (float, optional): sigma parameter. Defaults to 0.2.
        """
        self.dim = dim
        self.mu = np.full((dim,), mu)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """
        Reset the process to the average noise.
        """
        self.x = copy(self.mu)

    def sample(self) -> np.ndarray:
        """
        Returns a random noise sample.

        Returns:
            np.ndarray: The random noise of the process.
        """
        self.x += self.theta * (self.mu - self.x) + self.sigma * npr.randn(self.dim)
        return self.x


class DDPG(Agent):
    def __init__(self, actor: Model, critique: Model):
        """Deep Deterministic Policy Gradient implementation.

        Uses an Ornstein-Uhlenback process to sample the exploration noise.

        Args:
            actor (Model): DDPG actor
            critique (Model): DDPG critique
        """
        self.actor = actor
        self.t_actor = copy(actor)
        self.critique = critique
        self.t_critique = copy(critique)
        self.noise = OUNoise(1)

    def select_action(self, state: NDArray, noise: float = 0.0) -> float:
        """Select action based on epsilon-greedy policy.

        Args:
            states (NDArray): the passed state
            noise (float, optional): Scaling of the OU noise contribution.
              Defaults to 0.0.

        Returns:
            float: the action selected by the agent
        """
        action = self.actor(state) + self.noise.sample() * noise
        return action

    def optimize(
        self, loss: Any, opt1: Any, opt2: Any, batch: Batch, gamma: float
    ) -> float:
        """Train actor and critique network one-step on a minibatch.

        The optimization step follows the reference DDPG algorithm, i.e. the actor loss
        is calculated as the mean on the minibatch and the critique loss follows the
        typical dqn standard target q value update.

        Args:
            loss (Any): loss function to update the critique
            opt1 (Any): optimizer for actor network
            opt2 (Any): optimizer for critique network
            batch (Batch): minibatch with states, actions, next_states, rewards and
              terminal booleans
            gamma (float): discount factor

        Returns:
            float: loss value of the minibatch prediction of the critique.
        """
        states = np.array(batch.state, dtype="float32")
        actions = np.array(batch.action, dtype="float32")[..., np.newaxis]
        next_states = np.array(batch.next_state, dtype="float32")
        rewards = np.array(batch.reward, dtype="float32")
        terminals = np.array(batch.done, dtype="float32")

        next_action = self.t_actor(next_states)
        next_q = self.t_critique([next_states, next_action])

        target_q = rewards + gamma * (1.0 - terminals) * next_q
        loss_val = self.train_step(loss, opt1, opt2, states, actions, target_q)

        return loss_val

    @tf.function
    def train_step(self, loss, opt1, opt2, states, actions, target_q) -> float:
        # actor loss calculation and update
        with tf.GradientTape() as tape:
            action_preds = self.actor(states)

            qa_vals = self.critique([states, action_preds])

            actor_loss = -tf.math.reduce_mean(qa_vals)

        a_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        opt1.apply_gradients(zip(a_grads, self.actor.trainable_variables))

        # critique loss calculation and update
        with tf.GradientTape() as tape:
            q_vals = self.critique((states, actions))

            loss_val = loss(target_q, q_vals)

        c_grads = tape.gradient(loss_val, self.critique.trainable_variables)
        opt2.apply_gradients(zip(c_grads, self.critique.trainable_variables))

        return loss_val.numpy()

    def update_targets(self, tau: float = 0.999) -> None:
        """Soft weight update for the target actor and critique networks. The update
        rate is determined by the parameter tau, which describes the percentage of
        weights to keep on each update, also known as the interpolation rate.

        Args:
            tau (float, optional): interpolation rate. Defaults to 0.999.
        """
        # target actor update
        self.t_actor.set_weights(
            [
                (1.0 - tau) * w_dqn + tau * w_target
                for w_dqn, w_target in zip(
                    self.actor.get_weights(), self.t_actor.get_weights()
                )
            ]
        )

        # target critique update
        self.t_critique.set_weights(
            [
                (1.0 - tau) * w_dqn + tau * w_target
                for w_dqn, w_target in zip(
                    self.critique.get_weights(), self.t_critique.get_weights()
                )
            ]
        )
