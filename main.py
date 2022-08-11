import numpy as np
import pygame

from pong.agent import DDQN, DQN, DQNAgent, UserAgent
from pong.environment import Action, Ball, Environment, Field, Player, state2vec

pygame.init()

white = (255, 255, 255)
gray = (128, 128, 128)
black = (0, 0, 0)
red = (255, 0, 0)


def draw_player(p: Player, upscale=1.0):
    x = upscale * p.pos_x
    y = upscale * p.pos_y
    w = upscale * p.width
    h = upscale * p.height
    pygame.draw.rect(dis, white, [x, y, w, h])


def draw_ball(ball: Ball, upscale=1.0):
    pos = upscale * ball.pos
    rad = upscale * ball.radius
    pygame.draw.circle(dis, white, pos, rad)

def draw_field(field: Field, upscale=1.0):
    x_center = upscale * (field.origin[0] + 0.5 * field.width)
    dis.fill(black)
    pygame.draw.line(dis, gray, (x_center, upscale*field.origin[1]), (x_center, upscale*(field.origin[1] + field.height)))


WIDTH = 800
HEIGHT = 400
dis = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong by Kinguuu")

game_over = False
input_shape = 4

field = Field()
m2p = WIDTH / (field.width - field.origin[0])
env = Environment(field)

clock = pygame.time.Clock()

user_agent = UserAgent()
try:
    ai_agent1 = DDQN.load("models/ddqn1_new")
    ai_agent2 = DDQN.load("models/ddqn2_new")
except Exception:
    ai_agent1 = DQNAgent(DQN())
    ai_agent2 = DQNAgent(DQN())

action = Action.STILL
steps = 0


def flip_input(inputs):
    inputs[2] = field.width - inputs[2]
    inputs[4] = -inputs[4]
    inputs[5] = -inputs[5]

    return inputs


while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        action = user_agent.select_action(None, event)

    dis.fill(white)
    # Transform state tuple to input tensor
    current_state = env.observe()
    input_tensor1 = state2vec(current_state, target=0)[np.newaxis]
    input_tensor2 = state2vec(current_state, target=1)[np.newaxis]
    # input_tensor2 = flip_input(state2vec(current_state, target="opponent"))
    # input_tensor2 = np.expand_dims(input_tensor2, axis = 0)

    # action = ai_agent1.select_action(input_tensor1)

    env.act(action, ai_agent2.select_action(input_tensor2))
    draw_field(field, upscale=m2p)
    draw_player(env.p1, m2p)
    draw_player(env.p2, m2p)
    draw_ball(env.ball, m2p)

    pygame.display.update()

    clock.tick(60)
    steps += 1

    if steps % 1000 == 0:
        print(steps)

pygame.quit()
quit()
