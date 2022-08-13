import numpy as np
import random
from icecream import ic

from pong.agent import DDQN, DQN, UserAgent, SimpleAI
from pong.environment import Environment, Field, state2vec
from pong.renderer import Renderer

input_shape = 4

field = Field()
env = Environment(field)

renderer = Renderer(800, 400, env)

# Load user agents
user_agent = UserAgent()
simple_agent = SimpleAI(field, env.ball, env.p2)

try:
    ai_agent1 = DDQN.load("models/ddqn_single")
    ai_agent2 = DDQN.load("models/ddqn2_new")
    ic("Loaded")
except OSError as err:
    ai_agent1 = DDQN(DQN(), DQN())
    ai_agent2 = DDQN(DQN(), DQN())
    ic("Error")


def flip_input(inputs):
    inputs[2] = field.width - inputs[2]
    inputs[4] = -inputs[4]
    inputs[5] = -inputs[5]

    return inputs


user_control = False

random.seed(1111)
np.random.seed(1111)


while not renderer.game_over:
    action = renderer.events(user_control)

    # Transform state tuple to input tensor
    current_state = env.observe()
    input_tensor1 = state2vec(current_state, target=0)[np.newaxis, :]
    input_tensor2 = state2vec(current_state, target=1)[np.newaxis, :]

    if not user_control:
        action = ai_agent1.select_action(input_tensor1)

    #action2 = ai_agent2.select_action(input_tensor2)
    action2 = simple_agent.select_action(current_state)

    env.act(action, action2)

    renderer.render(120)

renderer.quit()
quit()
