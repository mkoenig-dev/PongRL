import random
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

State = namedtuple("State", ("agent", "opponent", "ball_pos", "ball_dir"))
Transition = namedtuple(
    "Transition",
    ("state", "action1", "action2", "new_state", "reward1", "reward2", "terminal"),
)

Batch = namedtuple(
    "Batch", ("states", "actions_indices", "new_states", "rewards", "terminal")
)

CollisionInfo = namedtuple(
    "CollisionInfo", ("collision", "left", "right", "bottom", "top")
)


class Action(Enum):
    """Defines actions for the pong environment"""

    DOWN = -1
    STILL = 0
    UP = 1


ACTION_SPACE = [Action.DOWN, Action.STILL, Action.UP]  # action space


class Reward(Enum):
    """Defines rewards for the pong environment"""

    SCORED = 2
    HIT = 1
    PLAYING = 0
    RECEIVED = -2


@dataclass
class Field:
    """Defines playing field parameters"""

    origin = np.zeros(2)
    width = 200.0
    height = 100.0
    speed = 1.0


@dataclass
class Player:
    """Defines player parameters"""

    width = 2.5
    height = 20.0
    pos_y = 0.0
    pos_x = 0.0
    hit = False
    player_speed = 0.7

    def reset(self, x: float, y: float):
        """Resets player to default at the kickoff.

        Args:
            x (float): reset position x
            y (float): reset position y
        """
        self.pos_x = x
        self.pos_y = y
        self.hit = False

    def update(self, action: Action, field: Field):
        """Updates player position based on an action.

        Args:
            action (Action): the action
            field (Field): the field
        """
        # update position
        self.pos_y += field.speed * action.value * self.player_speed

        # Clip players to field
        self.pos_y = max(
            min(field.origin[1] + field.height - self.height, self.pos_y),
            field.origin[1],
        )


@dataclass
class Ball:
    """Pong ball parameters"""

    pos = np.zeros(2)
    vel = np.zeros(2)
    radius = 2.0
    hit = 0
    wall_hit = 0
    last_touch = -1
    max_angle = np.deg2rad(55.0)
    ball_speed = 2.0
    kickoff_speed = 0.7

    def random_dir(self) -> NDArray:
        """Calculate a random kickoff direction.

        Returns:
            NDArray: the kickoff direction
        """
        theta = 0.5 * np.random.rand(1)[0] * np.pi - np.pi * 0.25
        vel = np.array([np.cos(theta), np.sin(theta)])
        sign = random.choice((-1.0, 1.0))

        return self.kickoff_speed * self.ball_speed * sign * vel

    def reset(self, x: float, y: float) -> None:
        """Resets ball parameters for kickoff.

        Args:
            x (float): reset x position
            y (float): reset y position
        """
        self.pos[0] = x
        self.pos[1] = y
        self.hit = 0
        self.wall_hit = 0
        self.last_touch = -1
        self.vel = self.random_dir()

    def intersect_wall(self, field: Field) -> None:
        """Update counter for wall intersections.

        Intersection occurred in this step iff wall_hit is 1 after the call.

        Args:
            field (Field): field instance for top and bottom wall intersection
        """
        if not (
            field.origin[1] < self.pos[1] - self.radius
            and self.pos[1] + self.radius < field.origin[1] + field.height
        ):
            self.wall_hit += 1
        else:
            self.wall_hit = 0

    def intersect_players(self, p1: Player, p2: Player) -> CollisionInfo:
        """Calculates player intersections with the ball.

        Updates the hit counter, where a collision only occured in this frame
        iff hit = 1.

        CollisionInfo contains the collision information, i.e. which boundary of the
        player rectange has been hit.

        Args:
            p1 (Player): player 1
            p2 (Player): player 2

        Returns:
            CollisionInfo: the collision info
        """
        c1 = intersect(self, p1)
        c2 = intersect(self, p2)

        if c1.collision:
            self.hit += 1
            p1.hit = True
            self.last_touch = 0
            current_collision = c1
        elif c2.collision:
            self.hit += 1
            p2.hit = True
            self.last_touch = 1
            current_collision = c2
        else:
            self.hit = 0
            p1.hit = False
            p2.hit = False
            current_collision = CollisionInfo(False, None, None, None, None)

        return current_collision

    def update(self, p1: Player, p2: Player, field: Field) -> None:
        """Updates ball position and velocity based on collisions.

        Args:
            p1 (Player): player 1
            p2 (Player): player 2
            field (Field): field instance
        """
        # Update ball position
        self.pos += self.vel * field.speed

        # Calculate intersections
        current_collision = self.intersect_players(p1, p2)
        self.intersect_wall(field)

        # Handle collisions with player
        if self.hit == 1:
            if current_collision.left is not None:
                bounce_angle = current_collision.left * self.max_angle
                self.vel[0] = self.ball_speed * -np.cos(bounce_angle)
                self.vel[1] = self.ball_speed * -np.sin(bounce_angle)
            elif current_collision.right is not None:
                bounce_angle = current_collision.right * self.max_angle
                self.vel[0] = self.ball_speed * np.cos(-bounce_angle)
                self.vel[1] = self.ball_speed * np.sin(-bounce_angle)
            else:
                # We do not need to check collision anymore
                self.vel[1] = -self.vel[1]

        # Handle collisions with wall
        if self.wall_hit == 1:
            self.vel[1] = -self.vel[1]


class GameState(Enum):
    """Current state of the game"""

    SCORED = 0
    PLAYING = 1
    RECEIVED = 2


def point_in_rectangle(point: NDArray, player: Player) -> bool:
    """Calculates whether the point is inside or outside the rectangle.

    Args:
        point (NDArray): 2D point
        player (Player): player for rectangle test

    Returns:
        bool: Whether point is inside player bounds.
    """
    vec_ap = point - np.array([player.pos_x, player.pos_y])
    vec_ab = np.array([player.width, 0])
    vec_ad = np.array([player.width, player.height])

    return 0 <= vec_ap.dot(vec_ab) <= vec_ab.dot(vec_ab) and 0 <= vec_ap.dot(
        vec_ad
    ) <= vec_ad.dot(vec_ad)


def calc_perpend(normal: NDArray, center: NDArray, point: NDArray) -> NDArray:
    """Calculate the projective point onto the ray defined by center and vec.

    Args:
        normal (NDArray): the normal of the line
        center (NDArray): point on the line
        point (NDArray): point to be projected

    Returns:
        NDArray: the projected point
    """

    dist = (point - center).dot(normal)
    return point - dist * normal, dist


def distance(vec1, vec2):
    """Returns the euclidean distance of the two vectors"""
    return np.linalg.norm(vec2 - vec1)


def distance2(vec1, vec2):
    """Returns the euclidean square distance between the two vectors"""
    diff = vec2 - vec1
    return diff.dot(diff)


def intersect_disc(ball: Ball, line: Tuple[NDArray, NDArray]) -> Optional[NDArray]:
    """Calculate the intersection point of the ball with a line

    Args:
        ball (Ball): the pong ball
        line (Tuple[NDArray, NDArray]): tuple of two vectors

    Returns:
        Optional[NDArray]: intersection point if there is an intersection,
          otherwise None.
    """
    vec = line[0] - line[1]
    normal = np.array([-vec[1], vec[0]])
    normal = normal / np.linalg.norm(normal)
    perp, dist = calc_perpend(normal, line[0], ball.pos)

    if abs(dist) < ball.radius and (
        distance(perp, line[0]) + distance(perp, line[1])
    ) == distance(line[0], line[1]):
        return perp
    if distance2(ball.pos, line[0]) < ball.radius**2:
        return line[0]
    if distance2(ball.pos, line[1]) < ball.radius**2:
        return line[1]

    return None


def intersect(ball: Ball, player: Player) -> CollisionInfo:
    """Calculates player intersection with the ball.

    CollisionInfo contains the collision information, i.e. which boundary of the
    player rectange has been hit.

    Args:
        player (Player): player

    Returns:
        CollisionInfo: the collision info
    """
    p_bl = np.array([player.pos_x, player.pos_y])
    p_br = np.array([player.pos_x + player.width, player.pos_y])
    p_tl = np.array([player.pos_x, player.pos_y + player.height])
    p_tr = np.array([player.pos_x + player.width, player.pos_y + player.height])

    pir = point_in_rectangle(ball.pos, player)
    inter_left = intersect_disc(ball, (p_bl, p_tl))
    inter_right = intersect_disc(ball, (p_br, p_tr))
    inter_bottom = intersect_disc(ball, (p_bl, p_br))
    inter_top = intersect_disc(ball, (p_tl, p_tr))

    center_y = player.pos_y + 0.5 * player.height

    # Determine normalized heifht
    rel_inter_left = (
        (inter_left[1] - center_y) / player.height
        if inter_left is not None
        else inter_left
    )
    rel_inter_right = (
        (inter_right[1] - center_y) / player.height
        if inter_right is not None
        else inter_right
    )

    info = (inter_left, inter_right, inter_bottom, inter_top)

    collision_info = CollisionInfo(
        any(pnt is not None for pnt in info) or pir,
        rel_inter_left,
        rel_inter_right,
        inter_bottom,
        inter_top,
    )

    return collision_info


class Environment:
    def __init__(self, field: Field) -> None:
        """Pong environment class.

        Initializes two players and a ball in the passed field.

        Args:
            field (Field): _description_
        """
        self.field = field
        self.ball = Ball()
        self.p1 = Player()
        self.p2 = Player()
        self.state = {}
        self.states = []
        self.restart()

    def reset(self) -> None:
        """Resets game to initial kickoff state without restarting the game"""
        self.p1.reset(
            self.field.origin[0],
            self.field.origin[1] + 0.5 * (self.field.height - self.p1.height),
        )
        self.p2.reset(
            self.field.origin[0] + self.field.width - self.p2.width,
            self.field.origin[1] + 0.5 * (self.field.height - self.p2.height),
        )
        self.ball.reset(
            self.field.origin[0] + 0.5 * self.field.width,
            self.field.origin[1] + 0.5 * self.field.height,
        )

    def restart(self) -> None:
        """Restarts the game, calls reset method internally and clears the list of past
        states including the score.
        """
        self.reset()
        self.state = {
            "agent": self.p1.pos_y,
            "opponent": self.p2.pos_y,
            "ball_pos": self.ball.pos,
            "ball_dir": self.ball.vel,
            "hit": self.ball.hit,
            "wall_hit": self.ball.wall_hit,
            "score": np.zeros(2),
            "game_state": GameState.PLAYING,
            "reward": None,
        }

        self.states = [self.state]

    def observe(self, i: int = -1) -> NDArray:
        """Returns the state of the game at frame i.

        Args:
            i (int, optional): state index. Defaults to -1.

        Returns:
            NDArray: state at frame i.
        """
        return self.normalize(State(*list(self.states[i].values())[:4]))

    def normalize(self, state: State) -> NDArray:
        """Transforms state to normalized array.

        Args:
            state (State): the state.

        Returns:
            NDArray: normalized array.
        """
        return np.array(
            [
                2.0 * (state.agent / self.field.height) - 1.0,
                2.0 * (state.opponent / self.field.height) - 1.0,
                2.0 * (state.ball_pos[0] / self.field.width) - 1.0,
                2.0 * (state.ball_pos[1] / self.field.height) - 1.0,
                state.ball_dir[0] / self.ball.ball_speed,
                state.ball_dir[1] / self.ball.ball_speed,
            ],
            dtype="float32",
        )

    def denormalize(self, state: NDArray) -> NDArray:
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

    def update_state(
        self, game_state: GameState, reward1: float, reward2: float
    ) -> None:
        """Updates the current game_state and appends to the list of states.

        Args:
            game_state (GameState): current gamestate
            reward1 (float): reward of player 1
            reward2 (float): reward of player 2
        """
        self.state = {
            "agent": self.p1.pos_y,
            "opponent": self.p2.pos_y,
            "ball_pos": self.ball.pos,
            "ball_dir": self.ball.vel,
            "hit": self.ball.hit,
            "wall_hit": self.ball.wall_hit,
            "score": self.state["score"],
            "game_state": game_state,
            "reward1": reward1,
            "reward2": reward2,
        }

        if game_state == GameState.SCORED:
            self.state["score"][0] += 1
        elif game_state == GameState.RECEIVED:
            self.state["score"][1] += 1

        self.states.append(self.state)

    def new_gamestate(self) -> GameState:
        """Determine the current gamestate.

        Returns:
            GameState: Scored, Received or Playing
        """
        if self.ball.pos[0] - self.ball.radius < self.field.origin[0]:
            game_state = GameState.RECEIVED
        elif (
            self.ball.pos[0] + self.ball.radius
            > self.field.origin[0] + self.field.width
        ):
            game_state = GameState.SCORED
        else:
            game_state = GameState.PLAYING

        return game_state

    def rewards(self, game_state: GameState) -> Tuple[float, float]:
        """Determine the rewards of the passed gamestate and current state.

        Args:
            game_state (GameState): the gamestate

        Returns:
            Tuple[float, float]: reward of player 1, reward of player 2
        """
        r1, r2 = Reward.PLAYING.value, Reward.PLAYING.value
        if game_state == GameState.SCORED:
            r2 += Reward.RECEIVED.value
            if self.ball.last_touch == 0:
                r1 += Reward.SCORED.value
        elif game_state == GameState.RECEIVED:
            r1 += Reward.RECEIVED.value
            if self.ball.last_touch == 1:
                r2 += Reward.SCORED.value

        if self.p1.hit:
            r1 += Reward.HIT.value

        if self.p2.hit:
            r2 += Reward.HIT.value

        return r1, r2

    def step(self, action1: Action, action2: Action) -> Tuple:
        """Steps the environment by actions of player 1 and player 2.

        Args:
            action1 (Action): action of player 1
            action2 (Action): action of player 2

        Returns:
            state (NDArray): state before the step
            action1 (Action): action of player 1
            action2 (Action): action of player 2
            next_state (NDArray): state after the step
            reward1 (float): reward of player 1
            reward2 (float): reward of player 2
            done (bool): Whether the episode is over
        """

        # Update player positions and clip
        self.p1.update(action1, self.field)
        self.p2.update(action2, self.field)

        # Update ball position and handle collisions
        self.ball.update(self.p1, self.p2, self.field)

        # Determine gamestate after this step
        game_state = self.new_gamestate()
        reward1, reward2 = self.rewards(game_state)

        # Reset game on score
        if game_state != GameState.PLAYING:
            self.reset()

        # Update environment state
        self.update_state(game_state, reward1, reward2)

        return (
            self.observe(-2),
            action1,
            action2,
            self.observe(),
            reward1,
            reward2,
            game_state != GameState.PLAYING,
        )


def state2vec(state: State, target: int = 0) -> NDArray:
    """Transforms state instance to state array. The target value determines the
    point of view.

    target = 0: PoV is player 1
    target = 1: PoV is player 2

    Args:
        state (State): the state to transform
        target (int, optional): point of view. Defaults to 0.

    Returns:
        NDArray: target = 0: (p1.height, p2.height, ball position, ball direction)
          target = 1: (p2.height, p1.height, ball position, ball direction)

    Raises:
        ValueError: if target is not in (0, 1).
    """
    if target == 0:
        return np.array([state.agent, state.opponent, *state.ball_pos, *state.ball_dir])
    if target == 1:
        return np.array([state.opponent, state.agent, *state.ball_pos, *state.ball_dir])

    raise ValueError(f"Passed target {target} is unsupported (choice(0, 1))")
