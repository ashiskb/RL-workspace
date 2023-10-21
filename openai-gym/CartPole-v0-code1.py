#!/Users/ashis/venv-directory/venv-p310-RL-workspace/bin/python
#Please make this python file executable and then run it without passing it to python interpreter
#as the the interpreter listed on the first line will be invoked. Good luck!
#$ chmod +x CartPole-v0-code1.py
#$ ./CartPole-v0-code1.py
# Or, simply run with `python` interpreter under the venv
# Output: a pygame window will be created and display the output of the program.
import gymnasium as gym
from tqdm import tqdm

#number of timesteps
n = 500

#Since we pass render_mode="human", you should see a window pop up rendering the environment.
env = gym.make("CartPole-v0", render_mode="human")

env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in tqdm(range(n)):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
        #break

env.close()


