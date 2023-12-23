![PongRL](https://github.com/mkoenig-dev/PongRL/assets/51786860/861a5810-3ba9-49da-b8c9-d50416f55f4e)

# Pong Reinforcement Learning

A TensorFlow-based project to make a deep learning model learn to play the game Pong based on the paper "[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)" by Deepmind (Mnih et al., 2019).

The game environment is implemented with pygame and reconstructs the behavior of the Atari game.

## Methods
- Deep Deterministic Policy-Gradient (DDPG) Methods
- Double Deep Q Networks (DDQN)
- Deep Q Networks (DQN)
- Experience Replay and Epsilon Greedy

## Installation and Usage

**Preferred Requirements:** Python 3.9

The package can be installed with hatch or a virtual environment in the project's root directory:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```
Run one of the experiments, e.g. DDQN:
```bash
python experiments/train_ddqn.py
```

Now you can locate the trained model and load it within ``main.py`` to see the performance versus a simple AI algorithm or play against the AI agent.
