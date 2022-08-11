from code import interact
import random
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Tuple
from xml.dom.expatbuilder import InternalSubsetExtractor
from icecream import ic
import pygame

import numpy as np

State = namedtuple("State", ("agent", "opponent", "ball_pos", "ball_dir"))
Transition = namedtuple(
    "Transition",
    ("state", "action1", "action2", "new_state", "reward1", "reward2", "terminal"),
)

Batch = namedtuple(
    "Batch", ("states", "actions_indices", "new_states", "rewards", "terminal")
)

CollisionInfo = namedtuple("CollisionInfo", ("collision", "left", "right", "bottom", "top"))


class Action(Enum):
    DOWN = -1
    STILL = 0
    UP = 1


actions = [Action.DOWN, Action.STILL, Action.UP]


class Reward(Enum):
    SCORED = 100
    HIT = 0
    PLAYING = 0
    RECEIVED = -100


@dataclass
class Field:
    origin = np.zeros(2)
    width = 200.0
    height = 100.0
    speed = 1.0


@dataclass
class Player:
    width = 2.5
    height = 20.0
    pos_y = 0.0
    pos_x = 0.0
    hit = False

    def reset(self, x, y):
        self.pos_x = x
        self.pos_y = y

    def out_of_bounds(self, field: Field):
        return (
            self.height + self.pos_y >= field.origin + field.height
            or self.pos_y <= field.origin
        )

    def update(self, action: Action, field: Field):
        # update position
        self.pos_y += field.speed * action.value

        # Clip players to field
        self.pos_y = max(
            min(
                field.origin[1] + field.height - self.height, self.pos_y
            ),
            field.origin[1],
        )


@dataclass
class Ball:
    pos = np.zeros(2)
    vel = np.zeros(2)
    radius = 2.0
    hit = 0
    wall_hit = 0
    last_touch = -1
    max_angle = np.deg2rad(55.0)
    ball_speed = 2.0

    def random_dir(self):
        theta = 0.5 * np.random.rand(1)[0] * np.pi - np.pi * 0.25
        vel = np.array([np.cos(theta), np.sin(theta)])
        sign = random.choice((-1.0, 1.0))

        return self.ball_speed * sign * vel

    def reset(self, x, y):
        self.pos[0] = x
        self.pos[1] = y
        self.hit = 0
        self.wall_hit = 0
        self.last_touch = -1
        self.vel = self.random_dir()

    def intersect_wall(self, field: Field):
        if not (field.origin[1] < self.pos[1] - self.radius and self.pos[1] + self.radius < field.origin[1] + field.height):
            self.wall_hit += 1
        else:
            self.wall_hit = 0

    def intersect_players(self, p1: Player, p2: Player):
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

    def update(self, p1: Player, p2: Player, field: Field):
        self.pos += self.vel * field.speed

        current_collision = self.intersect_players(p1, p2)
        self.intersect_wall(field)

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

        if self.wall_hit == 1:
            self.vel[1] = - self.vel[1]

class GameState(Enum):
    SCORED = 0
    PLAYING = 1
    RECEIVED = 2


def point_in_rectangle(point: np.ndarray, player: Player):
    vec_ap = point - np.array([player.pos_x, player.pos_y])
    vec_ab = np.array([player.width, 0])
    vec_ad = np.array([player.width, player.height])

    return 0 <= vec_ap.dot(vec_ab) <= vec_ab.dot(vec_ab) and 0 <= vec_ap.dot(
        vec_ad
    ) <= vec_ad.dot(vec_ad)


def calc_perpend(normal, center, point):
    """
    Calculate the projective point onto the ray defined by center and vec.
    :param norm: normal
    :param center:
    :param point:
    :return:
    """

    dist = (point - center).dot(normal)
    return point - dist * normal, dist


def distance(vec1, vec2):
    return np.linalg.norm(vec2 - vec1)

def distance2(vec1, vec2):
    diff = vec2 - vec1
    return diff.dot(diff)


def intersect_disc(ball: Ball, line: Tuple[np.ndarray, np.ndarray]):
    vec = line[0] - line[1]
    normal = np.array([-vec[1], vec[0]])
    normal = normal / np.linalg.norm(normal)
    perp, dist = calc_perpend(normal, line[0], ball.pos)

    if abs(dist) < ball.radius and (distance(perp, line[0]) + distance(perp, line[1])) == distance(line[0], line[1]):
        return perp
    elif distance2(ball.pos, line[0]) < ball.radius ** 2:
        return line[0]
    elif distance2(ball.pos, line[1]) < ball.radius ** 2:
        return line[1]
    else:
        return None


def intersect(ball: Ball, player: Player):
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
    rel_inter_left = (inter_left[1] - center_y) / player.height if inter_left is not None else inter_left
    rel_inter_right = (inter_right[1] - center_y) / player.height if inter_right is not None else inter_right

    info = (inter_left, inter_right, inter_bottom, inter_top)

    collision_info = CollisionInfo(
        any([pnt is not None for pnt in info]) or pir, 
        rel_inter_left, 
        rel_inter_right, 
        inter_bottom, 
        inter_top
    )

    return collision_info


def state2vec(state, target=0) -> np.ndarray:
    if target == 0:
        return np.array([state.agent, state.opponent, *state.ball_pos, *state.ball_dir])
    elif target == 1:
        return np.array([state.opponent, state.agent, *state.ball_pos, *state.ball_dir])


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
            "reward2": reward2,
        }

        if game_state == GameState.SCORED:
            self.state["score"][0] += 1
        elif game_state == GameState.RECEIVED:
            self.state["score"][1] += 1

        self.states.append(self.state)

    def new_gamestate(self):
        if self.ball.pos[0] - self.ball.radius < self.field.origin[0]:
            game_state = GameState.RECEIVED
        elif self.ball.pos[0] + self.ball.radius > self.field.origin[0] + self.field.width:
            game_state = GameState.SCORED
        else:
            game_state = GameState.PLAYING

        return game_state

    def rewards(self, game_state: GameState):
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

    def act(self, action1: Action, action2: Action) -> Transition:

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

        return Transition(
            self.observe(-2),
            action1,
            action2,
            self.observe(),
            reward1,
            reward2,
            game_state != GameState.PLAYING,
        )
