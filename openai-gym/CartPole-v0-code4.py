#!/Users/ashis/venv-directory/venv-p310-RL-workspace/bin/python
#Please make this python file executable and then run it without passing it to python interpreter
#as the the interpreter listed on the first line will be invoked. Good luck!
#$ chmod +x CartPole-v0-code4.py
#$ ./CartPole-v0-code4.py
#Or, simply run with `python` interpreter.
# References adopted: https://deepnote.com/@ken-e7bd/Intro-to-Q-learning-in-RL-e11e39d2-cebf-4552-8920-2db18aab3bd6
import gymnasium as gym
import math
import imageio.v2 as imageio
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import load, dump

#The CartPole-v0 environment with a learning agent
# Goal is to control the cart (i.e., platform) with a pole attached by its bottom prt.
# Trick: The pole tends to fall right or left and you would need to balance it by moving the cart to the right or left on every step.
#On average, this random agent takes 12 to 15 steps before the pole falls and the episode ends
#Most of the environments in Gym have a `reward boundary`, which is the average reward that the agent should gain during 100 consecutive eposides to solve the environment.
#For cartpole, the boundary is 195. That means, on average, the agent must hold the stick for 195 time steps or longer.
#So, our random agent's performance is extremely poor.

def discretize_state(state, env, buckets=(1,1,6,12)):
    """    
    The problem: The original states in this game are continuous, which does not work with the basic Q-learning algorithm as it expects discrete states. By the way, a slightly advanced Q-learning strategy can work with continuous state space with the help of approximation. Let's leave that strategy out of the scope of this course! Sorry. please enroll the "AI with Reinforcement Learning" course in Spring'23 with Dr. B. Purpose of this function is to discretize the continuous state space into buckets. 

    :param state: current state's observation which needs discretizing
    :type state: 4-D float array
    :param env: the cartpole environment
    :type env: environment object returned most likely from a gym.make() call.
    :param buckets: this will be used to discretize the original continuous states in this Cartpole example, defaults to (1,1,6,12)
    :type buckets: tuple, optional
    :return: The discretized state space in the given buckets
    :rtype: tuple
    """
    ## [position of cart, velocity of cart, angle of pole, rotation rate of pole]
    ## i) x-coordinate of the pole's center of mass (i.e., cart position), unit: m
    ## ii) Cart velocity [-inf, inf], unit: m/s
    ## iii) the pole's angle to the cart/platform. the pole angle in radians (1 radian = 57.295 degrees); 
    ## iv) the pole's angular velocity [-inf, inf], unit: radian/s
    

    # Revising the upper and the lower bounds for the discretization
    # Please note: cart velocity upper and lower bounds are 3.4e38 (inf), -3.4e38 (-inf). That's a huge space!
    # Let's shrink it down to [0.5, -0.5]

    # Also note: pole's angular velocity upper and lower bounds are 3.4e38 (inf), -3.4e38 (-inf). That's also a huge space!
    # Let's shrink it down to [50 degrees/1 sec, -50 degrees/1 sec]
    
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50) / 1.]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50) / 1.]

    # state is the native state representations produced by env
    ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
    
    # state_ is discretized state representation used for Q-table later
    state_ = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(state))]
    state_ = [min(buckets[i] - 1, max(0, state_[i])) for i in range(len(state))]

    return tuple(state_) 


def epsilon_greedy_policy(state, env, Q_table, exploration_rate):
    """This is an epsilon greedy policy. In other words, most of the times the agent chooses the action that maximizes the reward given state (greedily). But occassionally (controlled by the exploration_rate), the agent chooses a random action which makes sure the agent balances between exploitation and exploration

    :param state: the current state the agent is at.
    :type state: same as state
    :param env: the CartPole environment
    :type env: environment type returned perhaps from gym.make() call.
    :param Q_table: a table-like structure
    :type Q_table: same as Q_table
    :param exploration_rate: exploration rate
    :type exploration_rate: a small number close to 0.
    :return: action to be taken in the next step
    :rtype: between any value in the action_space. E.g., {0 (left), 1 (right)}
    """
    if (np.random.random() < exploration_rate):
        # Generates numbers np.random.random() uniformly between 0-1
        # This samples a random action given the environment
        return env.action_space.sample()
    else:
        # Choose greedily the action which gives the highest expected reward
        # given the current state
        return np.argmax(Q_table[state])


def get_rate_with_decay(t, decay_rate=25.0):
    """Get the learning rate or exploration_rate given an episode subject to decay. 
    Given the current episode number and the rate has a tendency to decrease with increasing number of episodes.

    :param t: episode number
    :type t: int
    :param decay_rate: decay rate, defaults to 25%
    :type decay_rate: float
    :return: decayed alpha value
    :rtype: float
    """
    decayed_alpha = max(0.1, min(1., 1. - np.log10((t + 1) / decay_rate)))
    return decayed_alpha



def update_Q(Q_table, state, action, reward, new_state, alpha, gamma):
    """Q-learning update step.

    :param Q_table: a table-like structure with N rows for states and M columns for actions
    :type Q_table: numpy array of shape (shape(discretized_state_space),shape(action_space)). Example: (1,1,6,12,2), where (1,1,6,12) is the shape of discretized state space, and (2,) is the shape of action space.
    :param state: the current state the agent is at time step t.
    :type state: numpy array of shape(discretized_state_space)
    :param action: the action taken given the previous state at time step t
    :type action: int
    :param reward: reward collected as a result of that action at time step t
    :type reward: int
    :param new_state: the new state at time-step t+1
    :type new_state: same as state
    :param alpha: learning rate
    :type alpha: float
    :param gamma: discount factor
    :type gamma: float
    :return: updated Q_table
    :rtype: same shape of the given Q_table
    """
    Q_table[state][action] = Q_table[state][action] + alpha * (reward + gamma * np.max(Q_table[new_state]) - Q_table[state][action])
    return Q_table



def Q_learning(env, num_episodes, gamma=0.98):
    """Training the agent with Q-learning with respect to pseudocode in Algorithm 1

    :param env: the cartpole environment
    :type env: environment object likely returned from a gym.make() call.
    :param num_episodes: the number of episodes for which to train
    :type num_episodes: int
    :param gamma: Discount factor gamma represents how much does the agent value future rewards as opposed to immediate rewards.
    :type gamma: float
    :return: The optimized Q-table
    :rtype: (dim(discretized_state_space)+dim(action_space)), e.g., (1,1,6,12,2) in CartPole-v0
    :return: A list containing the total cummulative reward for each episode of training.
    :rtype: list of length==num_episodes
    """

    # (1, 1, 6, 12) represents the discretization buckets.
    # Initialize the Q-table as full of zeros at the start.
    # Shape of Q_table would be = (1,1,6,12,  2), as there are 2 actions.
    Q_table = np.zeros((1, 1, 6, 12) + (env.action_space.n,))

    # Create a list to store the accumulated reward per each episode
    total_reward = []
    for e in tqdm(range(num_episodes)):

        # Reset the environment for a new episode, get the default state S_0
        state,info = env.reset()
        #convert the continuous state to discrete state
        state = discretize_state(state, env)

        # Adjust the alpha and the exploration rate, it is a coincidence they are the same.
        alpha = exploration_rate = get_rate_with_decay(e)
        
        # Initialize the current episode reward to 0 
        episode_reward = 0
        done = False
        while done is False:
            # Choose the action A_{t} based on the policy
            action = epsilon_greedy_policy(state, env, Q_table, exploration_rate)

            # Get the new state (S_{t+1}), reward (R_{t+1}), end signal
            new_state, reward, done, _, _ = env.step(action)
            #convert the continuous state to discrete state
            new_state = discretize_state(new_state, env)

            # Update Q-table via update_q(Q_table, S_{t}, A_{t}, R_{t+1}, S_{t+1}, alpha, gamma) 
            Q_table = update_Q(Q_table, state, action, reward, new_state, alpha, gamma)

            # Update the state S_{t} = S_{t+1}
            state = new_state
            
            # Accumulate the reward
            episode_reward += reward
        
        total_reward.append(episode_reward)
    print('Finished training!')
    return Q_table, total_reward


if __name__=='__main__':
    verbose = False
    do_training = False
    do_test = True
    do_debug = False

    if do_debug:
        ##Debug area begins
        pass
        ##Debug area ends

    if do_training:
        #Now, let's begin train
        # OpenAI Gym builds the environment for us including all the rules, dynamics etc.
        env = gym.make('CartPole-v0',render_mode='rgb_array')

        # How long do we want the agent to explore and learn?
        num_episodes = 1000      

        # Let us use Q-learning to learn best policy
        Q_table, total_reward = Q_learning(env, num_episodes)

        if verbose:
            #Plot
            plt.plot(range(num_episodes), total_reward)
            plt.xlabel('Episode')
            plt.ylabel('Training cumulative reward')
            plt.show()
            print(Q_table, Q_table.shape)

        #Saving the Q table for later use.
        dump(Q_table,'joblibs/Q_table.joblib')

        #Closing the gym environment
        env.close()
        
    if do_test:
        #Let's run one episode

        max_episode_length = 400 #Duh! You wish!! It depends on how good your agent learned during training. 

        # This way we can test the agent's recently learned policy with the saved Q_table
        Q_table = load('joblibs/Q_table.joblib')

        ##Final test and viz. Don't forget to switch to render_mode='rgb_array', otherwise env.render()
        ## will return None.
        env = gym.make('CartPole-v0',render_mode='rgb_array')

        # Initialize the reward
        episode_reward = 0

        # Count how many times the agent went right and how many times it went left
        right = 0
        left = 0

        # Initialize empty buffer for the images that will be stiched to a gif
        # Create a temp directory
        filenames = []
        try:
            os.mkdir("./temp")
        except:
            #print('Error: file system readonly?')
            pass

        # Test the trained agent in a completely fresh start environment
        state,info = env.reset()
        # Don't forget to discretize the state_space the same way you did to train the agent
        state = discretize_state(state, env)
        episode = 1
        step = 0
        # Run for maximum of max_episode_length steps which is the limit of the game
        for i in tqdm(range(max_episode_length)):

            # Plot the previous state and save it as an image that 
            # will be later patched together sa a .gif
            img = plt.imshow(env.render())

            plt.title("Episode: {}, Step: {}".format(episode,step))
            plt.axis('off')
            plt.savefig("./temp/{}.png".format(i))
            plt.close()
            filenames.append("./temp/{}.png".format(i))
            
            # Here we set the exploration rate to 0.0 as we want to avoid any random exploration.
            # That is, we want the agent fully depends on its learned policy (+Q_table)
            action = epsilon_greedy_policy(state, env, Q_table, exploration_rate=0.0)

            #Just for statistics purpose
            right+=1 if action == 1 else 0
            left+=1 if action == 0 else 0

            #Apply the next step
            new_state, reward, done, _ , _ = env.step(action)
            step += 1
            #Don't forget to discretize new_state
            new_state = discretize_state(new_state, env)
            state = new_state

            #Collect/accumulate reward
            episode_reward += reward

            # At the end of the episode print the total reward, 
            # only the agent is done before the set max_episode_length steps.
            # If your agent was trained well, who knows, the following would never happen! Haha
            if done:
                episode += 1
                step = 0
                print(f'Test episode finished at step {step+1} with a total reward of: {episode_reward}')
                print(f'We moved {right} times right and {left} times left')
                state,info = env.reset()
                state = discretize_state(state, env)
                #break
                
        # Stitch the images together to produce a .gif
        with imageio.get_writer('./video/CartPole-v0-QLearning.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        # Cleanup the images for the next run
        for f in filenames:
            os.remove(f)

        # Close the environment
        env.close()