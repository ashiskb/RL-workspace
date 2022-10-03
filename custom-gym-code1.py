#!/Users/ashis/venv-directory/venv-ml-p3.10/bin/python3.10
#Please make this python file executable and then run it without passing it to python interpreter
#as the the interpreter listed on the first line will be invoked. Good luck!
#$ chmod +x custom-gym-code1.py
#$ ./custom-gym-code1.py

import gym

#The Game
#We’re going to build a maze game that has 2 simple rules:
#The agent can move 1 step at a time in 4 directions: Up, Down, Left, and Right.
#The game is over when the agent reaches the goal.

#Custom class `ImageMazeEnv` must extend `gym.Env` class

# Then, you need to override 2 attributes and 4 methods which function as follow:
# Attributes:
#- action_space: All available actions the agent can perform.
#- observation_space: Structure of the observation.
#
#Methods:
#- step: Perform an action to the environment then return the state of the env, the reward of the action, and whether the episode is finished.
#- reset: Reset the state of the environment then return the initial state.
#- render(optional): Render the environment for visualization.
#- close(optional): Perform cleanup.
#Note that all the code related to this must be in an envs folder inside your project directory.

#Action & Observation space
#The action space is straightforward. There are 4 available actions: Left, Right, Up, and Down. We can define it using Discrete class provided for discrete space.

class ImageMazeEnv(gym.Env):
     def __init__(self):
         ...
     def step(self, action):
         ...
     def reset(self):
         ...
     def render(self):
         ...
     def close(self):
         ...