import os
import importlib
import gym
import numpy as np
import torch

from td3.args import get_args_eval

args = get_args_eval()

if args.custom_gym is not None and args.custom_gym != "":
    module = importlib.import_module(args.custom_gym, package=None)
    print("imported env '{}'".format((args.custom_gym)))

eval_env = gym.make(args.env_name)
eval_env.seed(args.seed + 100)

policy = torch.load(os.path.expanduser(args.policy), map_location=torch.device('cpu'))
policy.actor.eval()  # set to test mode

while True:
    state, done = eval_env.reset(), False
    eval_env.render()
    cum_reward = 0
    while not done:

        action = policy.select_action(np.array(state))
        state, reward, done, _ = eval_env.step(action)
        print (f"action {action}\treward {reward}")
        eval_env.render()
        cum_reward += reward
    print(f"Episode rewards: {cum_reward}")
