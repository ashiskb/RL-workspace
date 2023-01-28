"""Q-Learning demonstration!

Simple deterministic Markov Decision Process is made of 6 grids (states)

---------------------------------
|         |          |          |
|  Start  |          |  Goal    |
|         |          |          |
---------------------------------
|         |          |          |
|         |          |  Hole    |
|         |          |          |
---------------------------------

"""

from collections import deque
import numpy as np
import os
import time
from termcolor import colored
from joblib import load, dump


class QWorld():
    def __init__(self):
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
        self.reset()
        self.is_explore = True


    # save member variables in a pickle
    def save_q_world(self, joblib_file='joblibs/a_q_world.joblib'):
        #f = open(pkl_filename,'wb')
        #pickle.dump([self.col, self.row, self.q_table, self.gamma, self.epsilon,
        #             self.epsilon_decay, self.epsilon_min, self.is_explore],f)
        dump([self.col, self.row, self.q_table, self.gamma, self.epsilon, 
            self.epsilon_decay, self.epsilon_min, self.is_explore], joblib_file)
        #f.close()

    # load member variables from a pickle file
    def load_q_world(self, joblib_file = 'joblibs/a_q_world.joblib'):
        #f = open(pkl_filename,'rb')
        [self.col, self.row, self.q_table, self.gamma, self.epsilon,
                     self.epsilon_decay, self.epsilon_min, self.is_explore] = load(joblib_file)
        #f.close()

    # start of episode
    def reset(self):
        self.state = 0
        return self.state

    # agent wins when the goal is reached
    def is_in_win_state(self):
        return self.state == 2

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

    def init_transition_table(self):
        """ TT[state_{i},action] = state_{i+1}
        0 - Left, 1 - Down, 2 - Right, 3 - Up
        -------------
        | 0 | 1 | 2 |
        -------------
        | 3 | 4 | 5 |
        -------------
        """
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

    # determine the next action
    def act(self):
        # 0 - Left, 1 - Down, 2 - Right, 3 - Up

        # action is from exploration, by following the distribution "epsilon"
        if np.random.rand() <= self.epsilon:
            # explore - do random action
            self.is_explore = True
            return np.random.choice(4, 1)[0]

        # otherwise,  action is from exploitation
        # exploit - choose action with max Q-value
        self.is_explore = False

        return np.argmax(self.q_table[self.state])

    # Q-Learning - update the Q Table using Q(s, a)
    def update_q_table(self, state, action, reward, next_state):
        # Remember the Bellman equation to update the Q table?
        # Q(s, a) = reward + gamma * max_a' Q(s', a')
        q_value = reward + self.gamma * np.amax(self.q_table[next_state])

        self.q_table[state, action] = q_value

    # Print Q Table contents
    def print_q_table(self):
        print("Q-Table (Epsilon: %0.2f)" % self.epsilon)
        print(self.q_table)

    # update Exploration-Exploitation mix
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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
        #sys.exit("print_world")


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


# UI to display the world, delay of 1 sec for ease of understanding
def print_status(q_world, done, step, delay=1):
    os.system('clear')
    q_world.print_world(action, step)
    q_world.print_q_table()
    if done:
        print("-------EPISODE DONE--------")
        delay *= 2
    time.sleep(delay)


# main loop of Q-Learning
if __name__ == '__main__':
    print('Do you want to train the agent? ',end=' ')
    ans = input('[y/N] ').lower()
    if ans in ['yes', 'y']:
        training_flag = True
    elif ans in ['no', 'n','N']:
        training_flag = False
    else:
        print("Didn't understand your input! Guessing you do not want to train at all!")
        training_flag = False

    if training_flag == True:
        maxwins = 2000
        delay = 0
    else:
        maxwins = 10
        delay = 1


    wins = 0
    episode_count = 10 * maxwins
    # scores (max number of steps bef goal) - good indicator of learning
    scores = deque(maxlen=maxwins)
    q_world = QWorld()


    if training_flag == False:
        #Load pre-existing Q-table
        print("Loading q-world...")
        q_world.load_q_world()

    step = 1
    exit_flag = False
    # state, action, reward, next state iteration
    for episode in range(episode_count):
        state = q_world.reset()
        done = False
        print_episode(episode, delay=delay)
        while not done:
            action = q_world.act()
            next_state, reward, done = q_world.step(action)
            q_world.update_q_table(state, action, reward, next_state)
            print_status(q_world, done, step, delay=delay)
            state = next_state
            # if episode is done, perform housekeeping
            if done:
                if q_world.is_in_win_state():
                    wins += 1
                    scores.append(step)
                    if wins > maxwins:
                        exit_flag = True
                # Exploration-Exploitation is updated every episode
                q_world.update_epsilon()
                step = 1
            else:
                step += 1
            if exit_flag==True:
                break
        if exit_flag == True:
            break
    #print("Done..")

    if training_flag == True:
        print("Saving q-world for future use...")
        q_world.save_q_world()
    print(scores)
    q_world.print_q_table()



