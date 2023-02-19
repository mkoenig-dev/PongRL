import random
import sys

import numpy as np
import tensorflow as tf

from pong.agent import DDPG, DDQN, QModel, SimpleAI, UserAgent
from pong.environment import ACTION_SPACE, Environment, Field
from pong.renderer import Renderer

SEED = 121234129
USER_CONTROL = False

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

field = Field()
env = Environment(field)

renderer = Renderer(512, 256, env)

# Load user agents
user_agent = UserAgent()
simple_agent = SimpleAI(env, 1)

try:
    ai_agent1 = DDPG.load("models/ddpg")
    ai_agent2 = DDQN.load("models/ddqn2_new")
except OSError:
    ai_agent1 = DDQN(QModel((None, 6)))
    ai_agent2 = DDQN(QModel((None, 6)))


while not renderer.game_over:
    action2 = renderer.events(USER_CONTROL)

    # Transform state tuple to input tensor
    current_state = env.observe()

    action1 = ai_agent1.select_action(current_state[np.newaxis])[0][0]
    nearest_action_index = np.argmin(
        [abs(action.value - action1) for action in ACTION_SPACE], axis=0
    )

    nearest_action = ACTION_SPACE[np.squeeze(nearest_action_index)]

    # action2 = ai_agent2.select_action(input_tensor2)
    if not USER_CONTROL:
        action2 = simple_agent.select_action(current_state)

    env.step(nearest_action, action2)

    renderer.render(60)

renderer.quit()
sys.exit()
