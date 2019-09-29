import numpy as np
import torch
import gym
import os

from td3 import TD3, utils
from td3.args import get_args_train
from td3.utils import eval_policy

args = get_args_train()

file_name = f"td3_{args.env_name}_s{args.seed}"
print("---------------------------------------")
print(f"Settings: {file_name}")
print("---------------------------------------")

if not os.path.exists("../results"):
    os.makedirs("../results")

if not os.path.exists("../trained_models"):
    os.makedirs("../trained_models")

env = gym.make(args.env_name)

# Set seeds
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_action": max_action,
    "discount": args.discount,
    "tau": args.tau,
}

# Target policy smoothing is scaled wrt the action scale
kwargs["policy_noise"] = args.policy_noise * max_action
kwargs["noise_clip"] = args.noise_clip * max_action
kwargs["policy_freq"] = args.policy_freq
policy = TD3.TD3(**kwargs)

replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

# Evaluate untrained policy
evaluations = [eval_policy(policy, args.env_name, args.seed)]

state, done = env.reset(), False
episode_reward = 0
episode_timesteps = 0
episode_num = 0

for t in range(int(args.max_timesteps)):

    episode_timesteps += 1

    # Select action randomly or according to policy
    if t < args.start_timesteps:
        action = env.action_space.sample()
    else:
        action = (policy.select_action(np.array(state)) + np.random.normal(
            0, max_action * args.expl_noise, size=action_dim)).clip(
                -max_action, max_action)

    # Perform action
    next_state, reward, done, _ = env.step(action)
    done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

    # Store data in replay buffer
    replay_buffer.add(state, action, next_state, reward, done_bool)

    state = next_state
    episode_reward += reward

    # Train agent after collecting sufficient data
    if t >= args.batch_size:
        policy.train(replay_buffer, args.batch_size)

    if done:
        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        print(
            f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
        )
        # Reset environment
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # Evaluate episode
    if (t + 1) % args.eval_freq == 0:
        evaluations.append(eval_policy(policy, args.env_name, args.seed))
        np.save(f"../results/{file_name}", evaluations)
        torch.save(policy, f"../trained_models/{file_name}.pth")
