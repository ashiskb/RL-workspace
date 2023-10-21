#!/Users/ashis/venv-directory/venv-p310-RL-workspace/bin/python
#Please make this python file executable and then run it without passing it to python interpreter
#as the the interpreter listed on the first line will be invoked. Good luck!
#$ chmod +x CartPole-v0-code2.py
#$ ./CartPole-v0-code2.py
# Or, simply run with `python` interpreter under the venv

import gymnasium as gym
from tqdm import tqdm


#The CartPole-v0 environment
# Goal is to control the cart (i.e., platform) with a pole attached by its bottom prt.
# Trick: The pole tends to fall right or left and you would need to balance it by moving the cart to the right or left on every step.

env = gym.make("CartPole-v0")

#The observation of the environment is 4 floating point numbers:
## [position of cart, velocity of cart, angle of pole, rotation rate of pole]
## i) x-coordinate of the pole's center of mass
## ii) the pole's speed
## iii) the pole's angle to the cart/platform. the pole angle in radians (1 radian = 57.295 degrees)
## iv) the pole's rotation rate


obs,info = env.reset()
print('obs = {}'.format(obs))
#Example printout:
# obs = (array([0.01339712, 0.04502265, 0.03193145, 0.00526305], dtype=float32), {})


#The problem is: We need to convert hese 4 observations to actions. But, how do we learn to balance this system without knowing the exact meaning of the observed 4 numbers by getting the reward? 
# The reward is 1; it is given on every time step.
# The episode continues until the stick falls.
# To get a more accumulated reward, we need to balance the platform in a way to avoid the stick falling.

print('env.action_space = {}'.format(env.action_space))
#Example printout:
# env.action_space = Discrete(2)
#only 2 actions: 0 or 1, where 0 means pushing the platform to the left, 1 means to the right.


print('env.observation_space = {}'.format(env.observation_space))
#Example printout:
# env.observation_space = Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32)
#The observation space is a 4-D space, and each dimension is as follows:
#Num Observation             Min         Max
#0   Cart Position           -2.4        2.4
#1   Cart Velocity           -Inf        Inf
#2   Pole Angle              ~ -41.8°    ~ 41.8°
#3   Pole Velocity At Tip    -Inf        Inf
#env.observation_space.low and env.observation_space.high which will print the minimum and maximum values for each observation variable.
print('env.observation_space.high = {}'.format(env.observation_space.high))
print('env.observation_space.low = {}'.format(env.observation_space.low))
#Example printout:
#env.observation_space.high = [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
#env.observation_space.low = [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]


#Let's apply `action`
#How about going left, i.e., action=0 from the action_space
observation, reward, terminated, truncated, info = env.step(0)
print('observation = {}'.format(observation))
print('reward = {}'.format(reward))
print('terminated = {}'.format(terminated))
print('truncated = {}'.format(truncated))
print('info = {}'.format(info))
#Example printout:
#observation = [-0.02728556 -0.22667485 -0.01062018  0.3176722 ]
#reward = 1.0
#terminated = False
#truncated = False
#info = {}



#Let's apply random action with sampling
#The sample() returns a random sample from the underlying space.
#Here below you see we sample from the action_space
#The sample() can also be used in observation_space as well -- although why would we want to use that.
action = env.action_space.sample()
print('action = {}'.format(action))
#Example printout:
#action = 0

#Let's apply another random action with sampling
action = env.action_space.sample()
print('action = {}'.format(action))
#Example printout:
#action = 1

#Let's apply another random action with sampling
action = env.action_space.sample()
print('action = {}'.format(action))
#Example printout:
#action = 1

