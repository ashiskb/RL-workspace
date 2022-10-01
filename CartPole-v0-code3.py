#!/Users/ashis/venv-directory/venv-ml-p3.10/bin/python3.10
#Please make this python file executable and then run it without passing it to python interpreter
#as the the interpreter listed on the first line will be invoked. Good luck!
#$ chmod +x CartPole-v0-code3.py
#$ ./CartPole-v0-code3.py
import gym
from tqdm import tqdm


#The CartPole-v0 environment with a random agent
# Goal is to control the cart (i.e., platform) with a pole attached by its bottom prt.
# Trick: The pole tends to fall right or left and you would need to balance it by moving the cart to the right or left on every step.

env = gym.make("CartPole-v0",render_mode='human')

#Here below, we created the environment and initialized few variables.
total_reward = 0.0
total_steps = 0
observation, info = env.reset(seed=42)

while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    total_steps += 1

    if terminated:
        break

print('Episode terminated in {} steps\nTotal rewards accumulated = {}'.format(total_steps,total_reward))

#On average, this random agent takes 12 to 15 steps before the pole falls and the episode ends
#Most of the environments in Gym have a `reward boundary`, which is the average reward that the agent should gain during 100 consecutive eposides to solve the environment.
#For cartpole, the boundary is 195. That means, on average, the agent must hold the stick for 195 time steps or longer.
#So, our random agent's performance is extremely poor.