#!/Users/ashis/venv-directory/venv-ml-p3.10/bin/python3.10
# coding: utf-8

# # Non-gym environment
# ## Action-Environment Loop (From scratch)
# 

# # Non-Gym game development (Part 2)
# ## Goal vs Hole - v1
# * * Simple 2D text-based game with the following environment:
# 
# ```text
# --------------------------------------------
# |         |          |          |          |
# |  Start  |          |          |          |
# |         |          |          |          |
# --------------------------------------------
# |         |          |          |          |
# |         |          |  Hole    |          |
# |         |          |          |          |
# --------------------------------------------
# |         |          |          |          |
# |         |  Hole    |  Goal    |          |
# |         |          |          |          |
# --------------------------------------------
# |         |          |          |          |
# |         |          |          |  Hole    |
# |         |          |          |          |
# --------------------------------------------
# ```
# * The environment is a 4x4 grid.
# * 4 terminal states: one `Goal` and three `Hole` states; if a player moves into any of the four terminal states, the game is over.
# * 12 non-terminal states.
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
class Goal_vs_Hole_v1():
    pass


# In[ ]:


def init_reward_table(self):
    """
        0 - Left, 1 - Down, 2 - Right, 3 - Up
    --------------------------------------------
    |         |          |          |          |
    |  Start  |          |          |          |
    |    0    |    1     |    2     |    3     |
    --------------------------------------------
    |         |          |  -100    |          |
    |         |          |  Hole    |          |
    |    4    |    5     |    6     |    7     |
    --------------------------------------------
    |         |  -100    |  +100    |          |
    |         |  Hole    |  Goal    |          |
    |    8    |    9     |    10    |    11    |
    --------------------------------------------
    |         |          |          |  -100    |
    |         |          |          |  Hole    |
    |   12    |   13     |    14    |    15    |
    --------------------------------------------
    """
    
    self.reward_table = np.zeros([self.row, self.col])
    self.reward_table[5, 2] = -100.
    self.reward_table[5, 1] = -100.
    self.reward_table[2, 1] = -100.
    self.reward_table[7, 0] = -100.
    self.reward_table[8, 2] = -100.
    self.reward_table[11,0] = 100.
    self.reward_table[11,1] = -100.
    self.reward_table[13,3] = -100.
    self.reward_table[14,3] = 100.
    self.reward_table[14,2] = -100.

Goal_vs_Hole_v1.init_reward_table = init_reward_table


# In[ ]:


def init_transition_table(self):
    """ TT[state_{i},action] = state_{i+1}
        0 - Left, 1 - Down, 2 - Right, 3 - Up
        ------------------
        | 0 | 1 | 2   | 3 |
        -------------------
        | 4 | 5 | 6H  | 7 |
        -------------------
        | 8 | 9H | 10G| 11| 
        -------------------
        |12 | 13 | 14 |15H|
        -------------------
    """
    left, down, right, up = (0,1,2,3)
    self.action_space = [left,down,right,up]
    
    self.observation_space = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    self.observation_space_terminal = [6,9,10,15]
    self.observation_space_non_terminal = [0,1,2,3,4,5,7,8,11,12,13,14]
    
    
    self.transition_table = np.zeros([self.row, self.col], dtype=int)

    T = np.zeros_like(self.transition_table)
    #non-goal and non-hole transitions
    T[0,left], T[0,right], T[0,up], T[0, down] = (0,1,0,4)
    T[1,left], T[1,right], T[1,up], T[1, down] = (0,2,1,5)
    T[2,left], T[2,right], T[2,up], T[2, down] = (1,3,2,6)
    T[3,left], T[3,right], T[3,up], T[3, down] = (2,3,3,7)
    T[4,left], T[4,right], T[4,up], T[4, down] = (4,5,0,8)
    T[5,left], T[5,right], T[5,up], T[5, down] = (4,6,1,9)
    T[7,left], T[7,right], T[7,up], T[7, down] = (6,7,3,11)
    T[8,left], T[8,right], T[8,up], T[8, down] = (8,9,4,12)
    T[11,left], T[11,right], T[11,up], T[11, down] = (10,11,7,15)
    T[12,left], T[12,right], T[12,up], T[12, down] = (12,13,8,12)
    T[13,left], T[13,right], T[13,up], T[13, down] = (12,14,9,13)
    T[14,left], T[14,right], T[14,up], T[14, down] = (13,15,10,14)

    # terminal Goal state
    T[10,left], T[10,right], T[10,up], T[10,down] = (10,10,10,10)

    # terminal Hole state
    T[6,left], T[6,right], T[6,up], T[6, down] = (6,6,6,6)
    T[9,left], T[9,right], T[9,up], T[9, down] = (9,9,9,9)
    T[15,left], T[15,right], T[15,up], T[15, down] = (15,15,15,15)

    self.transition_table = T
    
Goal_vs_Hole_v1.init_transition_table = init_transition_table


# In[ ]:


# start of episode
def reset(self, start_state=0):
    self.state = start_state
    return self.state

Goal_vs_Hole_v1.reset = reset


# In[ ]:


def __init__(self, start_state=0):
    # 4 actions
    # 0 - Left, 1 - Down, 2 - Right, 3 - Up
    self.col = 4
    #self.action_space = (4,)

    # 16 states, 4x4 grid
    self.row = 16
    #self.observation_space = (4,4)

    # setup the environment
    self.q_table = np.zeros([self.row, self.col])
    self.init_transition_table()
    self.init_reward_table()

    # discount factor
    self.gamma = 0.9

    # 90% exploration, 10% exploitation
    self.epsilon = 0.9
    # exploration decays by this factor every episode
    self.epsilon_decay = 0.99
    # in the long run, 10% exploration, 90% exploitation
    # meaning, the agent is never going to bore you with predictable moves :D
    self.epsilon_min = 0.1

    # reset the environment
    self.reset()
    self.is_explore = True
    
    
Goal_vs_Hole_v1.__init__ = __init__


# In[ ]:


# agent wins when the goal is reached
def is_in_win_state(self):
    return self.state == 10
Goal_vs_Hole_v1.is_in_win_state = is_in_win_state


# In[ ]:


# execute the action on the environment
def step(self, action, verbose=False):
    if verbose:
        print('self.transition_table')
        print(self.transition_table)
    # determine the next_state given state and action
    next_state = self.transition_table[self.state, action]

    # done is True if next_state is Goal or Hole
    done = next_state == 6 or next_state == 9 or next_state == 10 or next_state == 15

    # reward given the state and action
    reward = self.reward_table[self.state, action]

    # the enviroment is now in new state
    self.state = next_state

    return next_state, reward, done

Goal_vs_Hole_v1.step = step


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
        return np.random.choice(valid_actions_from_state[0])

    # otherwise,  action is from exploitation
    # exploit - choose action with max Q-value
    self.is_explore = False

    return np.argmax(self.q_table[self.state])

Goal_vs_Hole_v1.act = act


# In[ ]:


# Q-Learning - update the Q Table using Q(s, a)
def update_q_table(self, state, action, reward, next_state):
    # Remember the Bellman equation to update the Q table?
    # Q(s, a) = reward + gamma * max_a' Q(s', a')
    q_value = reward + self.gamma * np.amax(self.q_table[next_state])

    self.q_table[state, action] = q_value
    
Goal_vs_Hole_v1.update_q_table = update_q_table


# In[ ]:


# update Exploration-Exploitation mix
def update_epsilon(self, do_test=False):
    if do_test:
        self.epsilon = 0.0 #absolutely no exploration during test time
    else:
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
Goal_vs_Hole_v1.update_epsilon = update_epsilon


# In[ ]:


def state_coord(self, state):
    return (state//4, 2+4*(state%4))
    if state==0: return (0,2)
    elif state==1: return (0,6)
    elif state==2: return (0,10)
    elif state==3: return (0,14)
    
Goal_vs_Hole_v1.state_coord = state_coord


# In[ ]:


# UI to display agent moving on the grid
# Playing with the terminal, so you could see "kinda" animation!
def print_cell(self, row=0):
    print("")
    for i in range(17): #col
        if self.state in [6,9,10,15]: #current state is one of the terminal states
            hole_cells = [(6,10,1),(9,6,2),(15,14,3)]
            goal_cells = [(10,10,2)]
            flag = False
            color_goal = 'green'
            if (self.state,i,row) in hole_cells:
                marker = "\033[4mH\033[0m"
                flag = True
            elif (self.state,i,row) in goal_cells:
                marker = "\033[4mG\033[0m"
                flag = True
            elif (i,row) in [(10,1),(6,2),(14,3)]:
                marker = 'H' #hole state
                flag= True
            elif (i,row) in [(10,2)]:
                marker = 'G' #goal state
                flag = True

            if i in [1,3,5,7,9,11,13,15] and flag==False: #none of the above gets printed
                marker = ' ' 
            elif i in [2,6,10,14] and flag==False:
                marker = ' '
            elif flag==False:
                marker = ''
                
            if (self.state,i,row) in goal_cells:
                color_goal = 'red'
            elif (self.state,i,row) in hole_cells:
                color_goal = 'blue'
            else:
                color_goal = 'green'
            print(colored(marker, color_goal), end='')
        else: #current state is a non-terminal state
            color_goal = 'green'
            H_G_flag = False
            r,c = self.state_coord(self.state)
            if (r,c) in [(row,i)]:
                marker = '_'
                color_goal = 'red'
                H_G_flag = True
            elif (i,row) in [(10,1),(6,2),(14,3)]:
                marker = 'H' #hole state
                H_G_flag = True
            elif (i,row) in [(10,2)]:
                marker = 'G' #goal state
                H_G_flag = True

            if i in [1,3,5,7,9,11,13,15] and H_G_flag==False: #none of the above gets printed
                marker = ' ' 
            elif i in [2,6,10,14] and H_G_flag==False:
                marker = ' '
            elif H_G_flag==False:
                marker = ''
            print(colored(marker, color_goal), end='')

        if i % 4 == 0: #every 4 col, close the cell with a boundary
            print('|', end='')
    print("")    
    
Goal_vs_Hole_v1.print_cell = print_cell


# In[ ]:


# UI to display mode and action of agent
def print_world(self, action, step):
    actions = {0: "(Left)", 1: "(Down)", 2: "(Right)", 3: "(Up)"}
    explore = "Explore" if self.is_explore else "Exploit"
    print("Step", step, ":", explore, actions[action])
    for _ in range(17):
        print('-', end='')
    self.print_cell() #row=0
    for _ in range(17):
        print('-', end='')
    for r in [1,2,3]: #remaining rows
        self.print_cell(row=r) #print row `r`
        for _ in range(17):
            print('-', end='')
    print("")

Goal_vs_Hole_v1.print_world = print_world


# In[ ]:


# Print Q Table contents
def print_q_table(self):
    print("Q-Table (Epsilon: %0.2f)" % self.epsilon)
    print(self.q_table)
Goal_vs_Hole_v1.print_q_table = print_q_table


# In[ ]:


# save member variables in a pickle
def save_world(self, joblib_file='joblibs/goal_vs_hole_world_v1.joblib'):
    dump([self.col, self.row, self.q_table, self.gamma, self.epsilon, 
        self.epsilon_decay, self.epsilon_min, self.is_explore], joblib_file)

Goal_vs_Hole_v1.save_world = save_world


# In[ ]:


# load member variables from a pickle file
def load_world(self, joblib_file = 'joblibs/goal_vs_hole_world_v1.joblib'):
    [self.col, self.row, self.q_table, self.gamma, self.epsilon,
            self.epsilon_decay, self.epsilon_min, self.is_explore] = load(joblib_file)
    
Goal_vs_Hole_v1.load_world = load_world


# In[ ]:


# UI to display episode count
def print_episode(episode, delay=1):
    os.system('clear')
    for _ in range(17):
        print('=', end='')
    print("")
    print("Episode ", episode)
    for _ in range(17):
        print('=', end='')
    print("")
    time.sleep(delay)


# In[ ]:


# UI to display the world, delay of 1 sec for ease of understanding
def print_status(the_world, done, step, delay=1,training_mode=True):
    os.system('clear')
    the_world.print_world(action, step)
    if training_mode: the_world.print_q_table()
    if done:
        print("-------EPISODE DONE--------")
        delay *= 2
    time.sleep(delay)


# In[ ]:


do_training = False 
do_test = True 


# In[ ]:


if do_training==True:
    maxwins = 100
    delay = 0
    wins = 0
    episode_count = 10 * maxwins
    # scores (max number of steps bef goal) - good indicator of learning
    scores = deque(maxlen=maxwins)
    goal_vs_hole_v1_world = Goal_vs_Hole_v1()
    
    
    step = 1
    exit_flag = False
    # state, action, reward, next state iteration
    for episode in range(episode_count):
        state = goal_vs_hole_v1_world.reset()
        done = False
        print_episode(episode, delay=delay)
        while not done:
            action = goal_vs_hole_v1_world.act()
            next_state, reward, done = goal_vs_hole_v1_world.step(action)
            goal_vs_hole_v1_world.update_q_table(state, action, reward, next_state)
            print_status(goal_vs_hole_v1_world, done, step, delay=delay)
            state = next_state
            # if episode is done, perform housekeeping
            if done:
                if goal_vs_hole_v1_world.is_in_win_state():
                    wins += 1
                    scores.append(step)
                    if wins > maxwins:
                        exit_flag = True
                # Exploration-Exploitation is updated every episode
                goal_vs_hole_v1_world.update_epsilon()
                step = 1
            else:
                step += 1
            if exit_flag==True:
                break
        if exit_flag == True:
            break
    #print("Done..")
    print("Saving world for future use...")
    goal_vs_hole_v1_world.save_world()
    print(scores)
    goal_vs_hole_v1_world.print_q_table()


# In[ ]:


if do_test == True:
    maxwins = 10
    delay = 1
    
    wins = 0
    episode_count = 10 * maxwins
    # scores (max number of steps bef goal) - good indicator of learning
    scores = deque(maxlen=maxwins)
    
    goal_vs_hole_v1_world = Goal_vs_Hole_v1(start_state=0)
    
    
    #Load pre-existing Q-table
    print("Loading a world...")
    goal_vs_hole_v1_world.load_world()

    step = 1
    exit_flag = False
    # state, action, reward, next state iteration
    for episode in range(episode_count):
        start_state = np.random.choice(goal_vs_hole_v1_world.observation_space_non_terminal)
        state = goal_vs_hole_v1_world.reset(start_state = start_state)
        done = False
        print_episode(episode, delay=delay)
        while not done:
            action = goal_vs_hole_v1_world.act()
            next_state, reward, done = goal_vs_hole_v1_world.step(action)
            goal_vs_hole_v1_world.update_q_table(state, action, reward, next_state)
            print_status(goal_vs_hole_v1_world, done, step, delay=delay)
            state = next_state
            # if episode is done, perform housekeeping
            if done:
                if goal_vs_hole_v1_world.is_in_win_state():
                    wins += 1
                    scores.append(step)
                    if wins > maxwins:
                        exit_flag = True
                # Exploration-Exploitation is updated every episode
                goal_vs_hole_v1_world.update_epsilon(do_test=do_test)
                step = 1
            else:
                step += 1
            if exit_flag==True:
                break
        if exit_flag == True:
            break
    #print("Done..")
    print(scores)
    goal_vs_hole_v1_world.print_q_table()


# # Thanks for your attention
