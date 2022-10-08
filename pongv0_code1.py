#!/Users/ashis/venv-directory/venv-ml-p3.10/bin/python3.10
#Please make this python file executable and then run it without passing it to python interpreter
#as the the interpreter listed on the first line will be invoked. Good luck!
#$ chmod +x pongv0_code1.py
#$ ./pongv0_code1.py

import gym
import math
import imageio.v2 as imageio
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import load, dump



#number of timestepts
n = 500

#Since we pass render_mode="human", you should see a window pop up rendering the environment.
render_mode = 'rgb_array' #or, 'human'
env = gym.make("ALE/Pong-v5", render_mode=render_mode)
# Initialize empty buffer for the images that will be stiched to a gif
# Create a temp directory
filenames = []
try:
    os.mkdir("./temp")
except:
    pass


env.action_space.seed(42)

observation, info = env.reset(seed=42)
episode=1
step=0
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
    if terminated or truncated:
        episode += 1
        step = 0
        observation, info = env.reset()
        #break

# Stitch the images together to produce a .gif
with imageio.get_writer('./video/ALE_Pong-v5-random.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Cleanup the images for the next run
for f in filenames:
    os.remove(f)


env.close()