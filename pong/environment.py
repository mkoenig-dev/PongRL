import random
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from collections import namedtuple

State = namedtuple("State",
                   ("agent", "opponent", "ball_pos", "ball_dir"))
Transition = namedtuple("Transition",
                        ("state", "action1", "action2", "new_state", "reward1", "reward2"))


class Action(Enum):
    DOWN = -1
    STILL = 0
    UP = 1


actions = [Action.DOWN, Action.STILL, Action.UP]


class Reward(Enum):
    SCORED = 100
    HIT = 50
    PLAYING = 1
    RECEIVED = -1000


@dataclass
class Ball:
    pos = np.zeros(2)
    vel = np.zeros(2)
    radius = 2.0
    hit = False
    wall_hit = False

    def reset(self, x, y):
        self.pos[0] = x
        self.pos[1] = y
        self.hit = False
        self.wall_hit = False

        theta = 0.5 * np.random.rand(1) * np.pi - np.pi * 0.25

        vel = np.array(
            np.cos(theta),
            np.sin(theta)
        )
        
        sign = random.choice((-1.0, 1.0))

        self.vel = sign * vel

    def mirror_x(self):
        if not self.hit:
            self.vel[0] = -self.vel[0]
        self.hit = True

    def mirror_y(self):
        if not self.wall_hit:
            self.vel[1] = -self.vel[1]
        self.wall_hit = True


@dataclass
class Player:
    width = 5.0
    height = 20.0
    pos_y = 0.0
    pos_x = 0.0
    hit = False

    def reset(self, x, y):
        self.pos_x = x
        self.pos_y = y

    def hit_ball(self, ball: Ball):
        hit = intersect(ball, self)
        return hit


class GameState(Enum):
    SCORED = 0
    PLAYING = 1
    RECEIVED = 2


@dataclass
class Field:
    origin = np.zeros(2)
    width = 200.0
    height = 100.0
    speed = 3.0

    def out_of_bounds(self, player: Player):
        return player.height + player.pos_y >= self.origin + self.height or \
               player.pos_y <= self.origin

    def wall_hit(self, ball: Ball):
        return not (self.origin[1] < ball.pos[1] < self.origin[1] + self.height)


def point_in_rectangle(point: np.ndarray, player: Player):
    vec_ap = point - np.array([player.pos_x, player.pos_y])
    vec_ab = np.array([player.width, 0])
    vec_ad = np.array([player.width, player.height])

    return 0 <= vec_ap.dot(vec_ab) <= vec_ab.dot(vec_ab) and \
           0 <= vec_ap.dot(vec_ad) <= vec_ad.dot(vec_ad)


def calc_perpend(norm, center, point):
    """
    Calculate the projective point onto the ray defined by center and vec.
    :param norm: normal
    :param center:
    :param point:
    :return:
    """

    dist = (point - center).dot(norm)
    return point - dist * norm, dist


def distance(vec1, vec2):
    return np.linalg.norm(vec2 - vec1)


def intersect_disc(ball: Ball, line: Tuple[np.ndarray, np.ndarray]):
    vec = line[1] - line[0]
    norm = np.array([
        -vec[1],
        vec[0]
    ])
    perp, dist = calc_perpend(norm, line[0], ball.pos)

    return dist <= ball.radius and \
           distance(perp, line[0]) + distance(perp, line[1]) == distance(line[0], line[1])


def intersect(ball: Ball, player: Player):
    p_bl = np.array([
        player.pos_x,
        player.pos_y
    ])
    p_br = np.array([
        player.pos_x + player.width,
        player.pos_y
    ])
    p_tl = np.array([
        player.pos_x,
        player.pos_y + player.height
    ])
    p_tr = np.array([
        player.pos_x + player.width,
        player.pos_y + player.height
    ])

    return (point_in_rectangle(ball.pos, player)) or \
           intersect_disc(ball, (p_bl, p_br)) or \
           intersect_disc(ball, (p_bl, p_tl)) or \
           intersect_disc(ball, (p_br, p_tr)) or \
           intersect_disc(ball, (p_tl, p_tr))


def state2vec(state, target="agent") -> np.ndarray:
    if target == "agent":
        return np.array([
            state.agent,
            state.opponent,
            *state.ball_pos,
            *state.ball_dir
        ])
    elif target == "opponent":
        return np.array([
            state.opponent,
            state.agent,
            *state.ball_pos,
            *state.ball_dir
        ])


class Environment:
    def __init__(self, field: Field):
        self.field = field
        self.ball = Ball()
        self.p1 = Player()
        self.p2 = Player()
        self.state = {}
        self.states = []
        self.restart()

    def reset(self):
        self.p1.reset(
            self.field.origin[0],
            self.field.origin[1] + 0.5 * (self.field.height - self.p1.height)
        )
        self.p2.reset(
            self.field.origin[0] + self.field.width - self.p2.width,
            self.field.origin[1] + 0.5 * (self.field.height - self.p2.height)
        )
        self.ball.reset(
            self.field.origin[0] + 0.5 * self.field.width,
            self.field.origin[1] + 0.5 * self.field.height
        )

    def restart(self):
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

    def observe(self, i=-1) -> State:
        return State(*list(self.states[i].values())[:4])

    def handle_collisions(self):
        # TODO: Clean up later
        if self.p1.hit_ball(self.ball) or self.p2.hit_ball(self.ball):
            self.ball.mirror_x()
        else:
            self.ball.hit = False

        if self.field.wall_hit(self.ball):
            self.ball.mirror_y()
        else:
            self.ball.wall_hit = False

    def update_state(self, game_state, reward1, reward2):
        self.state = {
            "agent": self.p1.pos_y,
            "opponent": self.p2.pos_y,
            "ball_pos": self.ball.pos,
            "ball_dir": self.ball.vel,
            "hit": self.ball.hit,
            "wall_hit": self.ball.wall_hit,
            "score": np.zeros(2),
            "game_state": game_state,
            "reward1": reward1,
            "reward2": reward2
        }

        if game_state == GameState.SCORED:
            self.state["score"][0] += 1
        elif game_state == GameState.RECEIVED:
            self.state["score"][1] += 1

        self.states.append(self.state)

    def new_gamestate(self):
        if self.ball.pos[0] < self.field.origin[0]:
            game_state = GameState.RECEIVED
        elif self.ball.pos[0] > self.field.origin[0] + self.field.width:
            game_state = GameState.SCORED
        else:
            game_state = GameState.PLAYING

        return game_state

    def rewards(self, game_state: GameState):
        if game_state == GameState.SCORED:
            r1, r2 = Reward.SCORED, Reward.RECEIVED
        elif game_state == GameState.RECEIVED:
            r1, r2 = Reward.RECEIVED, Reward.SCORED
        else:
            r1, r2 = Reward.PLAYING, Reward.PLAYING

        if self.p1.hit:
            r1 += r1.HIT

        if self.p2.hit:
            r2 += r2.HIT

        return r1, r2

    def act(self, action1: Action, action2: Action) -> Transition:

        # Update player positions
        self.p1.pos_y += self.field.speed * action1.value
        self.p2.pos_y += self.field.speed * action2.value

        # Clip players to field
        self.p1.pos_y = max(
            min(
                self.field.origin[1] + self.field.height - self.p1.height,
                self.p1.pos_y
            ),
            self.field.origin[1]
        )

        self.p2.pos_y = max(
            min(
                self.field.origin[1] + self.field.height - self.p2.height,
                self.p2.pos_y
            ),
            self.field.origin[1]
        )

        # Handle ball collisions
        self.handle_collisions()

        # Update ball position
        self.ball.pos += self.ball.vel * self.field.speed

        # Determine gamestate after this step
        game_state = self.new_gamestate()
        reward1, reward2 = self.rewards(game_state)

        # Update environment state
        self.update_state(game_state, reward1, reward2)

        # Reset game on score
        if game_state != GameState.PLAYING:
            self.reset()

        return Transition(
            self.observe(-2),
            action1,
            action2,
            self.observe(),
            reward1,
            reward2
        )
