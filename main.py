import numpy as np

from pong.agent import DDQN, DQN, UserAgent
from pong.environment import Environment, Field, state2vec
from pong.renderer import Renderer
import tensorflow as tf


input_shape = 4

field = Field()
env = Environment(field)

renderer = Renderer(800, 400, env)

# Load user agents
user_agent = UserAgent()
try:
    ai_agent1 = DDQN.load("models/ddqn1_new")
    ai_agent2 = DDQN.load("models/ddqn2_new")
except Exception as err:
    print(err)
    ai_agent1 = DDQN(DQN(), DQN())
    ai_agent2 = DDQN(DQN(), DQN())


def flip_input(inputs):
    inputs[2] = field.width - inputs[2]
    inputs[4] = -inputs[4]
    inputs[5] = -inputs[5]

    return inputs


user_control = True


while not renderer.game_over:
    action = renderer.events(user_control)

    # Transform state tuple to input tensor
    current_state = env.observe()
    input_tensor1 = state2vec(current_state, target=0)[np.newaxis, :]
    input_tensor2 = state2vec(current_state, target=1)[np.newaxis, :]

    if not user_control:
        action = ai_agent1.select_action(input_tensor1)

    env.act(action, ai_agent2.select_action(input_tensor2))
    
    renderer.render(90)

renderer.quit()
quit()
