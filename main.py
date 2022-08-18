import random
import sys

import numpy as np
import tensorflow as tf

from pong.agent import DDQN, QModel, SimpleAI, UserAgent
from pong.environment import Environment, Field
from pong.renderer import Renderer

SEED = 121234129
USER_CONTROL = False

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

field = Field()
env = Environment(field)

renderer = Renderer(800, 400, env)

# Load user agents
user_agent = UserAgent()
simple_agent = SimpleAI(env, 1)

try:
    ai_agent1 = DDQN.load("models/dqn")
    ai_agent2 = DDQN.load("models/ddqn2_new")
except OSError:
    ai_agent1 = DDQN(QModel((None, 6)))
    ai_agent2 = DDQN(QModel((None, 6)))


while not renderer.game_over:
    action = renderer.events(USER_CONTROL)

    # Transform state tuple to input tensor
    current_state = env.observe()

    if not USER_CONTROL:
        action = ai_agent1.select_action(current_state[np.newaxis])

    # action2 = ai_agent2.select_action(input_tensor2)
    action2 = simple_agent.select_action(current_state)

    env.step(action, action2)

    renderer.render(120)

renderer.quit()
sys.exit()
