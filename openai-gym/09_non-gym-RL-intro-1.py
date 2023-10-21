#!/Users/ashis/venv-directory/venv-p310-RL-workspace/bin/python
# coding: utf-8

# # Non-gym environment
# ## Action-Environment Loop (From scratch)
# 

# # Non-Gym game development (Part 1)
# ## Goal vs Hole - v0
# * Simple 2D text-based game with the following environment:
# 
# ```text
# ---------------------------------
# |         |          |          |
# |  Start  |          |  Goal    |
# |         |          |          |
# ---------------------------------
# |         |          |          |
# |         |          |  Hole    |
# |         |          |          |
# ---------------------------------
# ```
# * The environment is a 2x3 grid.
# * 2 terminal states: `Goal` and `Hole`; if a player moves into any of the two terminal states, the game is over.
# * 4 non-terminal states.
# * **Rewards**
#     * 0 if on a non-terminal state,
#     * -100 if on `Hole` state,
#     * +100 if on `Goal` state.

# ### The imports

# In[ ]:


#the imports
from collections import deque
import numpy as np
import os
import time
from termcolor import colored
from joblib import load, dump


# In[ ]:


#Let's define a class
class Goal_vs_Hole_v0():
    pass


# In[ ]:


def init_reward_table(self):
    """
    0 - Left, 1 - Down, 2 - Right, 3 - Up
    ----------------
    | 0 | 0 | 100  |
    ----------------
    | 0 | 0 | -100 |
    ----------------
    """
    self.reward_table = np.zeros([self.row, self.col])
    self.reward_table[1, 2] = 100.
    self.reward_table[4, 2] = -100.

Goal_vs_Hole_v0.init_reward_table = init_reward_table


# In[ ]:


def init_transition_table(self):
    """ TT[state_{i},action] = state_{i+1}
    0 - Left, 1 - Down, 2 - Right, 3 - Up
    -------------
    | 0 | 1 | 2 |
    -------------
    | 3 | 4 | 5 |
    -------------
    """
    self.action_space = [0,1,2,3]
    self.observation_space = [0,1,2,3,4,5]
    self.observation_space_terminal = [2,5]
    self.observation_space_non_terminal = [0,1,3,4]
    
    self.transition_table = np.zeros([self.row, self.col], dtype=int)

    self.transition_table[0, 0] = 0
    self.transition_table[0, 1] = 3
    self.transition_table[0, 2] = 1
    self.transition_table[0, 3] = 0

    self.transition_table[1, 0] = 0
    self.transition_table[1, 1] = 4
    self.transition_table[1, 2] = 2
    self.transition_table[1, 3] = 1

    # terminal Goal state
    self.transition_table[2, 0] = 2
    self.transition_table[2, 1] = 2
    self.transition_table[2, 2] = 2
    self.transition_table[2, 3] = 2

    self.transition_table[3, 0] = 3
    self.transition_table[3, 1] = 3
    self.transition_table[3, 2] = 4
    self.transition_table[3, 3] = 0

    self.transition_table[4, 0] = 3
    self.transition_table[4, 1] = 4
    self.transition_table[4, 2] = 5
    self.transition_table[4, 3] = 1

    # terminal Hole state
    self.transition_table[5, 0] = 5
    self.transition_table[5, 1] = 5
    self.transition_table[5, 2] = 5
    self.transition_table[5, 3] = 5
    
Goal_vs_Hole_v0.init_transition_table = init_transition_table


# In[ ]:


# start of episode
def reset(self, start_state=0):
    self.state = start_state
    return self.state

Goal_vs_Hole_v0.reset = reset


# In[ ]:


def __init__(self, start_state=0):
    # 4 actions
    # 0 - Left, 1 - Down, 2 - Right, 3 - Up
    self.col = 4

    # 6 states
    self.row = 6

    # setup the environment
    self.q_table = np.zeros([self.row, self.col])
    self.init_transition_table()
    self.init_reward_table()

    # discount factor
    self.gamma = 0.9

    # 90% exploration, 10% exploitation
    self.epsilon = 0.9
    
    # exploration decays by this factor every episode
    self.epsilon_decay = 0.9
    # in the long run, 10% exploration, 90% exploitation
    # meaning, the agent is never going to bore you with predictable moves :D
    self.epsilon_min = 0.1

    # reset the environment
    self.reset(start_state)
    self.is_explore = True
    
    
Goal_vs_Hole_v0.__init__ = __init__


# In[ ]:


# agent wins when the goal is reached
def is_in_win_state(self):
    return self.state == 2
Goal_vs_Hole_v0.is_in_win_state = is_in_win_state


# In[ ]:


# execute the action on the environment
def step(self, action):
    # determine the next_state given state and action
    next_state = self.transition_table[self.state, action]

    # done is True if next_state is Goal or Hole
    done = next_state == 2 or next_state == 5

    # reward given the state and action
    reward = self.reward_table[self.state, action]

    # the enviroment is now in new state
    self.state = next_state

    return next_state, reward, done

Goal_vs_Hole_v0.step = step


# In[ ]:


# determine the next action
def act(self):
    # 0 - Left, 1 - Down, 2 - Right, 3 - Up

    # action is from exploration, by following the distribution "epsilon"
    if np.random.rand() <= self.epsilon:
        # explore - do random action
        self.is_explore = True
        
        #find valid transitions from current state
        valid_actions_from_state = np.where(self.transition_table[self.state,:]!=self.state)
        #return np.random.choice(4, 1)[0]  #4C1
        return np.random.choice(valid_actions_from_state[0])
        
    # otherwise,  action is from exploitation
    # exploit - choose action with max Q-value
    self.is_explore = False

    return np.argmax(self.q_table[self.state])

Goal_vs_Hole_v0.act = act


# In[ ]:


# Q-Learning - update the Q Table using Q(s, a)
def update_q_table(self, state, action, reward, next_state):
    # Remember the Bellman equation to update the Q table?
    # Q(s, a) = reward + gamma * max_a' Q(s', a')
    q_value = reward + self.gamma * np.amax(self.q_table[next_state])

    self.q_table[state, action] = q_value
    
Goal_vs_Hole_v0.update_q_table = update_q_table


# In[ ]:


# update Exploration-Exploitation mix
def update_epsilon(self, do_test=False):
    if do_test:
        self.epsilon = 0.0 #absolutely no exploration during test time
    else:
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
Goal_vs_Hole_v0.update_epsilon = update_epsilon


# In[ ]:


# UI to display agent moving on the grid
# Playing with the terminal, so you could see "kinda" animation!
def print_cell(self, row=0):
    print("")
    for i in range(13):
        j = i - 2
        if j in [0, 4, 8]:
            if j == 8:
                if self.state == 2 and row == 0:
                    marker = "\033[4mG\033[0m"
                elif self.state == 5 and row == 1:
                    marker = "\033[4mH\033[0m"
                else:
                    marker = 'G' if row == 0 else 'H'
                color = self.state == 2 and row == 0
                color = color or (self.state == 5 and row == 1)
                color = 'red' if color else 'blue'
                print(colored(marker, color), end='')
            elif self.state in [0, 1, 3, 4]:
                cell = [(0, 0, 0), (1, 0, 4), (3, 1, 0), (4, 1, 4)]
                marker = '_' if (self.state, row, j) in cell else ' '
                print(colored(marker, 'red'), end='')
            else:
                print(' ', end='')
        elif i % 4 == 0:
            print('|', end='')
        else:
            print(' ', end='')
    print("")
    
Goal_vs_Hole_v0.print_cell = print_cell


# In[ ]:


# UI to display mode and action of agent
def print_world(self, action, step):
    actions = {0: "(Left)", 1: "(Down)", 2: "(Right)", 3: "(Up)"}
    explore = "Explore" if self.is_explore else "Exploit"
    print("Step", step, ":", explore, actions[action])
    for _ in range(13):
        print('-', end='')
    self.print_cell()
    for _ in range(13):
        print('-', end='')
    self.print_cell(row=1)
    for _ in range(13):
        print('-', end='')
    print("")

Goal_vs_Hole_v0.print_world = print_world


# In[ ]:


# Print Q Table contents
def print_q_table(self):
    print("Q-Table (Epsilon: %0.2f)" % self.epsilon)
    print(self.q_table)
Goal_vs_Hole_v0.print_q_table = print_q_table


# In[ ]:


# save member variables in a pickle
def save_world(self, joblib_file='joblibs/goal_vs_hole_world_v0.joblib'):
    dump([self.col, self.row, self.q_table, self.gamma, self.epsilon, 
        self.epsilon_decay, self.epsilon_min, self.is_explore], joblib_file)

Goal_vs_Hole_v0.save_world = save_world


# In[ ]:


# load member variables from a pickle file
def load_world(self, joblib_file = 'joblibs/goal_vs_hole_world_v0.joblib'):
    [self.col, self.row, self.q_table, self.gamma, self.epsilon,
            self.epsilon_decay, self.epsilon_min, self.is_explore] = load(joblib_file)
    
Goal_vs_Hole_v0.load_world = load_world


# In[ ]:


# UI to display episode count
def print_episode(episode, delay=1):
    os.system('clear')
    for _ in range(13):
        print('=', end='')
    print("")
    print("Episode ", episode)
    for _ in range(13):
        print('=', end='')
    print("")
    time.sleep(delay)


# In[ ]:


# UI to display the world, delay of 1 sec for ease of understanding
def print_status(the_world, done, step, delay=1):
    os.system('clear')
    the_world.print_world(action, step)
    the_world.print_q_table()
    if done:
        print("-------EPISODE DONE--------")
        delay *= 2
    time.sleep(delay)


# In[ ]:


do_training = False 
do_test = True 


# In[ ]:


if do_training==True:
    maxwins = 2000
    delay = 0
    wins = 0
    episode_count = 10 * maxwins
    # scores (max number of steps bef goal) - good indicator of learning
    scores = deque(maxlen=maxwins)
    goal_vs_hole_v0_world = Goal_vs_Hole_v0()
    
    
    step = 1
    exit_flag = False
    # state, action, reward, next state iteration
    for episode in range(episode_count):
        state = goal_vs_hole_v0_world.reset()
        done = False
        print_episode(episode, delay=delay)
        while not done:
            action = goal_vs_hole_v0_world.act()
            next_state, reward, done = goal_vs_hole_v0_world.step(action)
            goal_vs_hole_v0_world.update_q_table(state, action, reward, next_state)
            print_status(goal_vs_hole_v0_world, done, step, delay=delay)
            state = next_state
            # if episode is done, perform housekeeping
            if done:
                if goal_vs_hole_v0_world.is_in_win_state():
                    wins += 1
                    scores.append(step)
                    if wins > maxwins:
                        exit_flag = True
                # Exploration-Exploitation is updated every episode
                goal_vs_hole_v0_world.update_epsilon()
                step = 1
            else:
                step += 1
            if exit_flag==True:
                break
        if exit_flag == True:
            break
    #print("Done..")
    print("Saving world for future use...")
    goal_vs_hole_v0_world.save_world()
    print(scores)
    goal_vs_hole_v0_world.print_q_table()


# In[ ]:


if do_test == True:
    maxwins = 10
    delay = 1
    
    wins = 0
    episode_count = 10 * maxwins
    # scores (max number of steps bef goal) - good indicator of learning
    scores = deque(maxlen=maxwins)
    
    goal_vs_hole_v0_world = Goal_vs_Hole_v0(start_state=0)
    
    
    #Load pre-existing Q-table
    print("Loading a world...")
    goal_vs_hole_v0_world.load_world()

    step = 1
    exit_flag = False
    # state, action, reward, next state iteration
    for episode in range(episode_count):
        start_state = np.random.choice(goal_vs_hole_v0_world.observation_space_non_terminal)
        state = goal_vs_hole_v0_world.reset(start_state = start_state)
        done = False
        print_episode(episode, delay=delay)
        while not done:
            action = goal_vs_hole_v0_world.act()
            next_state, reward, done = goal_vs_hole_v0_world.step(action)
            goal_vs_hole_v0_world.update_q_table(state, action, reward, next_state)
            print_status(goal_vs_hole_v0_world, done, step, delay=delay)
            state = next_state
            # if episode is done, perform housekeeping
            if done:
                if goal_vs_hole_v0_world.is_in_win_state():
                    wins += 1
                    scores.append(step)
                    if wins > maxwins:
                        exit_flag = True
                # Exploration-Exploitation is updated every episode
                goal_vs_hole_v0_world.update_epsilon(do_test=do_test)
                step = 1
            else:
                step += 1
            if exit_flag==True:
                break
        if exit_flag == True:
            break
    #print("Done..")
    print(scores)
    goal_vs_hole_v0_world.print_q_table()


# # Thanks for your attention
