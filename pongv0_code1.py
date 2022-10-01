#!/Users/ashis/venv-directory/venv-ml-p3.10/bin/python3.10
#Please make this python file executable and then run it without passing it to python interpreter
#as the the interpreter listed on the first line will be invoked. Good luck!
#$ chmod +x pongv0_code1.py
#$ ./pongv0_code1.py

import gym
from tqdm import tqdm

#number of timestepts
n = 500

#Since we pass render_mode="human", you should see a window pop up rendering the environment.
env = gym.make("Pong-v0", render_mode="human")
env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in tqdm(range(n)):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
        #break

env.close()