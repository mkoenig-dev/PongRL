import random
from abc import ABC, abstractmethod
from collections import deque
from copy import copy
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import numpy.random as npr
import pygame
import tensorflow as tf
from numpy.typing import ArrayLike
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Model

from .environment import ACTION_SPACE, Action, Batch, Environment, Transition


class ReplayBuffer:
    def __init__(self, size: int):
        """A replay buffer for experience replay training.

        Args:
            size (int): maximum length of the buffer, i.e. samples to store.
        """
        self.size = size
        self.memory = deque([], size)

    def push(self, *args) -> None:
        """Store an experience tuple as a transition.
        The arguments are converted to a transition.

        Args:
            *args: state, action1, action2, new_state, reward1, reward2, terminal
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        """Randomly sample a minibatch of transitions.

        Args:
            batch_size (int): minibatch size (smaller than len)

        Returns:
            List[Transition]: new list of sampled transitions
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Returns the number of experiences stored in the buffer."""
        return len(self.memory)


def epsilon_decay(
    e: int, num_episodes: int, eps_start: int, eps_end: int, warm_up: bool = False
) -> float:
    """Epsilon decay over episodes with warmup phase.

    If warmup is specified, the decay gets ignored and epsilon is set to the highest
    probability in a epsilon-greedy approach. Otherwise depending on the passed episode
    epsilon is interpolated in [0, num_episodes] to the range (eps_start, eps_end).

    Interpolated epsilon is clipped before return, so e > num_episodes returns eps_end.

    Args:
        e (int): current episode
        num_episodes (int): number of episodes to decay to eps_end
        eps_start (int): starting epsilon
        eps_end (int): epsilon at the end of the decay
        warm_up (bool, optional): 100% random action if True. Defaults to False.

    Returns:
        float: _description_
    """
    if warm_up:
        return 0.0

    return np.clip(
        (eps_end - eps_start) * (e / (num_episodes - 1)) + eps_start,
        eps_start,
        eps_end,
    )


class QModel(Model):
    def __init__(self, shape: ArrayLike):
        """Simple multilayer perceptron implementation for policy/target network.

        Args:
            shape (ArrayLike): input shape
        """
        super().__init__()
        self.input_layer = InputLayer(shape)
        self.x1 = Dense(256, activation="relu")
        self.x2 = Dense(128, activation="relu")
        self.out = Dense(3, activation="linear")

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.x1(x)
        x = self.x2(x)
        out = self.out(x)

        return out


class Agent(ABC):
    @abstractmethod
    def select_action(self, state: ArrayLike) -> Action:
        """Select an environment action based on the passed state.

        Args:
            state (ArrayLike): the passed state

        Returns:
            Action: the action selected by the agent
        """


class EpsilonGreedyAgent(Agent):
    def __init__(self, actor: Model):
        """Epsilon-greedy agent for keras models / action-value function approximations.

        Args:
            actor (Model): the q-value approimation model.
        """
        self.actor = actor

    def select_action(self, state: ArrayLike, eps: Optional[float] = None) -> Action:
        """Select action based on epsilon-greedy policy.

        Args:
            state (ArrayLike): the passed state
            eps (float, optional): Probability for random action. Defaults to None.

        Returns:
            Action: the action selected by the agent
        """
        if eps is not None and random.random() > eps:
            return random.choice(ACTION_SPACE)

        q_approx = self.actor(state)
        i = np.argmax(q_approx)

        return ACTION_SPACE[i]


class DDQN(Agent):
    def __init__(self, model: Model) -> None:
        """Double Deep Q-Network implementation.

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
        """Trains policy network in one step on a minibatch using a Double DQN update.

        Args:
            batch (Batch): minibatch with states, actions, next_states, rewards and
              terminal booleans
            gamma (float): discount factor

        Returns:
            float: loss value of the minibatch prediction.
        """
        if self.optimizer is None or self.loss is None:
            raise ValueError("Please compile the agent before training.")

        # Unpack batch
        states = batch.state
        actions = batch.action
        next_states = batch.next_state
        rewards = batch.reward
        terminals = batch.done

        loss_value = self._step(states, actions, next_states, rewards, terminals, gamma)

        return loss_value

    @tf.function
    def _step(
        self,
        states: Any,
        actions: Any,
        next_states: Any,
        rewards: Any,
        terminals: Any,
        gamma: float,
    ) -> float:
        """Optimization step for policy update. Outsourced to its own function
        to make use of tensorflow autograph features.

        Args:
            states (Any): transition states
            actions (Any): transition actions
            next_states (Any): transition next states
            rewards (Any): transition rewards
            terminals (Any): transition dones
            gamma (float): discount factor

        Returns:
            float: loss value of the minibatch prediction.
        """
        with tf.GradientTape() as tape:
            # Calculate target Q value for chosen actions

            q_values = tf.math.reduce_sum(
                self.policy(states) * tf.one_hot(actions, 3), axis=1
            )

            next_q_vals = self.policy(next_states)
            next_actions = tf.argmax(next_q_vals, axis=1)

            future_q = tf.math.reduce_sum(
                self.target(next_states) * tf.one_hot(next_actions, 3), axis=1
            )

            target_q = rewards + gamma * future_q * (1.0 - terminals)

            # Calculate Huber loss
            loss_value = self.loss(q_values, target_q)

        # Get gradients w.r.t. weights
        grads = tape.gradient(loss_value, self.policy.trainable_variables)
        clipped_grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]

        # Update policy network
        self.optimizer.apply_gradients(
            zip(clipped_grads, self.policy.trainable_variables)
        )

        return loss_value

    def select_action(self, state: ArrayLike, eps: Optional[float] = None) -> int:
        """Select action based on epsilon-greedy policy.

        Args:
            state (ArrayLike): the passed state
            eps (float, optional): Probability for random action. Defaults to None.

        Returns:
            Action: the action selected by the agent
        """
        if eps is not None and random.random() > eps:
            return random.choice(ACTION_SPACE)

        q_approx = self.policy(state)
        i = tf.squeeze(tf.argmax(q_approx, axis=1))

        return ACTION_SPACE[i]

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
        except Exception as exc:
            raise RuntimeError(f"Could not save model to {path}") from exc

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
            except Exception as exc:
                raise RuntimeError(f"Unable to load DDQN from {path}") from exc
        else:
            raise OSError("One of the paths for loading does not exist.")


class SimpleAI(Agent):
    def __init__(self, env: Environment, target: int = 0) -> None:
        """Simple Pong AI that predicts intersections of the ball on the goal plane.

        Args:
            env (Environment): pong environment
            target (int): the player to control (p1 = 0, p2 = 1)
        """
        self.env = env
        self.field = env.field
        self.ball = env.ball
        self.target = target
        self.player = env.p1 if target == 0 else env.p2
        self.agent = "agent" if target == 0 else "opponent"
        self.last_action = Action.STILL
        self.real_target_pos = np.zeros(2)
        self.target_pos = np.zeros(2)

    def predict_intersection(self, state: ArrayLike, depth: int = 0) -> np.ndarray:
        """Predicts the intersection point of the ball trajectory with the AI's
        goal plane from the passed state.

        Args:
            state (ArrayLike): the state for prediction
            depth (int, optional): Number of bounces to track. Defaults to 0.

        Returns:
            np.ndarray: the intersection point if the calculation resolves in the number
              of recursive bounce calls (depth). Otherwise the target position is the
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

            ball_pos += self.field.speed * ball_dir * ysteps
            ball_dir[1] = -ball_dir[1]

            # Recursive call on new position
            next_state = np.array([state[0], state[1], *ball_pos, *ball_dir])
            return self.predict_intersection(next_state, depth=depth + 1)

        return np.array([hrange[self.target], 0.5 * vrange.sum()])

    def select_action(self, state: ArrayLike) -> Action:
        state_t = self.env.denormalize(state)

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

    def select_action(self, state: Optional[ArrayLike], event=None) -> Action:
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

    def compile(self, loss: Any, opt_actor: Any, opt_critique: Any) -> None:
        self.loss = loss
        self.opt_actor = opt_actor
        self.opt_critique = opt_critique

    def select_action(self, state: ArrayLike, noise: float = 0.0) -> float:
        """Select action based on epsilon-greedy policy.

        Args:
            state (ArrayLike): the passed state
            noise (float, optional): Scaling of the OU noise contribution.
              Defaults to 0.0.

        Returns:
            float: the action selected by the agent
        """
        action = self.actor(state) + self.noise.sample() * noise
        return action

    def optimize(self, batch: Batch, gamma: float) -> float:
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
        states = np.asarray(batch.state, dtype="float32")
        actions = np.asarray(batch.action, dtype="float32")[..., np.newaxis]
        next_states = np.asarray(batch.next_state, dtype="float32")
        rewards = np.asarray(batch.reward, dtype="float32")
        terminals = np.asarray(batch.done, dtype="float32")

        next_action = self.t_actor(next_states)
        next_q = self.t_critique([next_states, next_action])

        target_q = rewards + gamma * (1.0 - terminals) * next_q
        loss_val = self._step(states, actions, target_q)

        return loss_val.numpy()

    @tf.function
    def _step(self, states, actions, target_q) -> float:
        # actor loss calculation and update
        with tf.GradientTape() as tape:
            action_preds = self.actor(states)

            qa_vals = self.critique([states, action_preds])

            actor_loss = -tf.math.reduce_mean(qa_vals)

        a_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.opt_actor.apply_gradients(zip(a_grads, self.actor.trainable_variables))

        # critique loss calculation and update
        with tf.GradientTape() as tape:
            q_vals = self.critique((states, actions))

            loss_val = self.loss(target_q, q_vals)

        c_grads = tape.gradient(loss_val, self.critique.trainable_variables)

        self.opt_critique.apply_gradients(
            zip(c_grads, self.critique.trainable_variables)
        )

        return loss_val

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

    def save(self, path: str):
        """Saves ddpg model to file using tf-model saving.
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
            self.actor.save(path / "actor")
            self.t_actor.save(path / "target_actor")
            self.critique.save(path / "critique")
            self.t_critique.save(path / "target_critique")
        except Exception as exc:
            raise RuntimeError(f"Could not save model to {path}") from exc

    def load(path: str) -> "DDPG":
        """Loads saved DDPG from specified model folder.
        Path must have two sub folders named "actor" and "critique" containing the model
        files.

        Args:
            path (str): saved model directory

        Raises:
            RuntimeError: If DDPG could not be loaded
            OSError: If one of the sub model files do not exist

        Returns:
            DDPG: new instance created from the loaded models
        """
        load_path = Path(path)
        actor_path = load_path / "actor"
        t_actor_path = load_path / "target_actor"
        critique_path = load_path / "critique"
        t_critique_path = load_path / "target_critique"

        paths = [actor_path, t_actor_path, critique_path, t_critique_path]

        if load_path.exists() and all(path_.exists() for path_ in paths):
            try:
                actor = tf.keras.models.load_model(actor_path)
                critique = tf.keras.models.load_model(critique_path)

                t_actor = tf.keras.models.load_model(t_actor_path)
                t_critique = tf.keras.models.load_model(t_critique_path)

                ddpg = DDPG(actor, critique)
                ddpg.t_actor = t_actor
                ddpg.t_critique = t_critique

                return ddpg
            except Exception as exc:
                raise RuntimeError(f"Unable to load DDPG from {path}") from exc
        else:
            raise OSError("One of the paths for loading does not exist.")
