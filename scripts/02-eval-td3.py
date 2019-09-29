import os

import gym
import numpy as np
import torch

from td3.args import get_args_eval

args = get_args_eval()

eval_env = gym.make(args.env_name)
eval_env.seed(args.seed + 100)

policy = torch.load(os.path.expanduser(args.policy))
policy.actor.eval()  # set to test mode

while True:
    state, done = eval_env.reset(), False
    eval_env.render()
    cum_reward = 0
    while not done:

        action = policy.select_action(np.array(state))
        state, reward, done, _ = eval_env.step(action)
        eval_env.render()
        cum_reward += reward
    print(f"Episode rewards: {cum_reward}")
