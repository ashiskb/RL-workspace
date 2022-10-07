# Learning Reinforcement Learning with Openai-gym
It's fun to work with Openai-gym and reinforcement learning. In this repository you are going to find a notebook and several python scripts to introduce to you only the basics.

It is expected that you follow the order below inspecting the codebase:

* `openai-gym-intro.ipynb`: a jupyter notebook describing openai-gym. It includes description of the following python scripts. It also includes Q-learning based reinforcement learning for the Cartpole-v0 gym environment.

## Please run the following scripts from the terminal as Jupyter notebook above may render gyms poorly.
* ALE-Breakout-v5_code1.py : The atari Breakout powered by ALE. The agent acts randomly per step.
* Blackjack-v1-code1.py : Blackjack gym. The agent acts randomly per step.
* CarRacing-v2-code1.py : CarRacing gym. The agent acts randomly per step.
* CartPole-v0-code1.py : CartPole gym. The agent acts randomly per step.
* CartPole-v0-code2.py : Basics / introduction to the CartPole-v0 gym.
* CartPole-v0-code3.py : Random agent for Cartpole-v0 gym.
* CartPole-v0-code4.py : Training an agent to play CartPole-v0 gym with Q-learning.
* Riverraid-v0_code1.py : Riverraid-v0 gym. The agent acts randomly per step.
* lunarlanderv2.py : LunarLanderv2 gym. The agent acts randomly per step.
* pongv0_code1.py : pongv0 gym. The agent acts randomly per step.

## Listing all environments in gym
* list_all_envs_registry.py :


## Requirements
* Python 3.10.x
* `pip install -r requirements.txt` (the file is given)
* Make sure you have `gym==0.26.1` installed. And verify `print(gym.__version__)` returns `0.26.1`.

