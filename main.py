import pygame
from pong.environment import Environment, Field, Player, Ball, Action, state2vec
from pong.agent import UserAgent, DQNAgent, DQN
import tensorflow as tf
import numpy as np

pygame.init()

white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)


def draw_player(p: Player, upscale=1.0):
    x = upscale * p.pos_x
    y = upscale * p.pos_y
    w = upscale * p.width
    h = upscale * p.height
    pygame.draw.rect(dis, black, [x, y, w, h])


def draw_ball(ball: Ball, upscale=1.0):
    pos = upscale * ball.pos
    rad = upscale * ball.radius
    pygame.draw.circle(dis, black, pos, rad)


WIDTH = 800
HEIGHT = 400
dis = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Pong by Kinguuu')

game_over = False
input_shape = 4

field = Field()
m2p = WIDTH / (field.width - field.origin[0])
env = Environment(field)

clock = pygame.time.Clock()

user_agent = UserAgent()
try:
    dqn1 = tf.keras.models.load_model("models/target1")
    dqn2 = tf.keras.models.load_model("models/target2")
except OSError:
    dqn1 = DQN()
    dqn2 = DQN()

ai_agent1 = DQNAgent(dqn1)
ai_agent2 = DQNAgent(dqn2)
action = Action.STILL

while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        # action = user_agent.select_action(None, event)

    dis.fill(white)
    # Transform state tuple to input tensor
    current_state = env.observe()
    input_tensor1 = np.expand_dims(state2vec(current_state), axis=0)
    input_tensor2 = np.expand_dims(state2vec(current_state, target="opponent"), axis=0)

    env.act(ai_agent1.select_action(input_tensor1), ai_agent2.select_action(input_tensor2))
    draw_player(env.p1, m2p)
    draw_player(env.p2, m2p)
    draw_ball(env.ball, m2p)

    pygame.display.update()

    clock.tick(60)

pygame.quit()
quit()
