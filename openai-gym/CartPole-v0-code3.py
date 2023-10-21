#!/Users/ashis/venv-directory/venv-p310-RL-workspace/bin/python
#Please make this python file executable and then run it without passing it to python interpreter
#as the the interpreter listed on the first line will be invoked. Good luck!
#$ chmod +x CartPole-v0-code3.py
#$ ./CartPole-v0-code3.py
# # Or, simply run with `python` interpreter under the venv
import gymnasium as gym
import math
import imageio.v2 as imageio
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import load, dump

#number of timesteps
n = 500



#The CartPole-v0 environment with a random agent
# Goal is to control the cart (i.e., platform) with a pole attached by its bottom prt.
# Trick: The pole tends to fall right or left and you would need to balance it by moving the cart to the right or left on every step.
render_mode = 'rgb_array' #or, 'human'
env = gym.make("CartPole-v0", render_mode=render_mode)
# Initialize empty buffer for the images that will be stiched to a gif
# Create a temp directory
filenames = []
try:
    os.mkdir("./temp")
except:
    pass

#Here below, we created the environment and initialized few variables.
total_reward = 0.0
total_steps = 0
observation, info = env.reset(seed=42)

episode = 1
step = 0
for i in tqdm(range(n)):
    # Plot the previous state and save it as an image that 
    # will be later patched together sa a .gif
    img = plt.imshow(env.render())

    plt.title("Episode: {}, Step: {}".format(episode,step))
    plt.axis('off')
    plt.savefig("./temp/{}.png".format(i))
    plt.close()
    filenames.append("./temp/{}.png".format(i))

    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    step += 1
    total_reward += reward
    total_steps += 1

    if terminated:
        episode += 1
        step = 0
        observation, info = env.reset()
        #break

#print('Episode terminated in {} steps\nTotal rewards accumulated = {}'.format(total_steps,total_reward))
# Stitch the images together to produce a .gif
with imageio.get_writer('./video/CartPole-v0-random.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Cleanup the images for the next run
for f in filenames:
    os.remove(f)


env.close()
#On average, this random agent takes 12 to 15 steps before the pole falls and the episode ends
#Most of the environments in Gym have a `reward boundary`, which is the average reward that the agent should gain during 100 consecutive eposides to solve the environment.
#For cartpole, the boundary is 195. That means, on average, the agent must hold the stick for 195 time steps or longer.
#So, our random agent's performance is extremely poor.