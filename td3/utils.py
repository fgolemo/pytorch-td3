from gym import Wrapper
import matplotlib.pyplot as plt
try:
    from comet_ml import Experiment
except ImportError:
    pass

import gym
import numpy as np
import torch

def start_comet(args):
    exp = None
    if args.comet is not None and len(args.comet) > 0:
        workspace, project, apikey = args.comet.split("/")
        exp = Experiment(api_key=apikey, project_name=project, workspace=workspace)
        exp.set_name("td3")
        if len(args.comet_tags) > 0:
            comet_tags = args.comet_tags.split(",")
            for tag in comet_tags:
                exp.add_tag(tag)
    return exp

class ReplayBuffer(object):

    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        #TODO if it's image, then make this uint
        if len(state_dim) == 1: # raw state, float enc
            self.state = np.zeros((max_size, state_dim), dtype=np.float32)
            self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        elif len(state_dim) == 3: # img, int enc
            self.state = np.zeros((max_size, *state_dim), dtype=np.uint8)
            self.next_state = np.zeros((max_size, *state_dim), dtype=np.uint8)
            print (self.state.shape)

        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (torch.FloatTensor(self.state[ind]).permute(0,3,1,2).div(255).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).permute(0,3,1,2).div(255).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device))

from collections import deque
import time

class Monitor(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env)
        self.episode_return = None
        self.episode_len = None
        self.episode_count = 0
        self.tstart = time.time()

    def reset(self):
        obs = self.env.reset()
        self.episode_return = 0
        self.episode_len = 0
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        self.episode_return += rew
        self.episode_len += 1

        if done:
            info['episode'] = {'r': self.episode_return, 'l': self.episode_len, 't': round(time.time() - self.tstart, 6)}
            self.episode_count += 1
            self.episode_return = 0
            self.episode_len = 0

        return obs, rew, done, info

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()

# Runs policy for X episodes and returns average reward
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

