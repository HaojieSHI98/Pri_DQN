#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import pickle
from collections import deque
from datetime import datetime

import gym
import numpy as np
import paddle.fluid as fluid
from tqdm import tqdm

import parl
from atari_agent import AtariAgent
from atari_model import AtariModel
from parl.utils import logger, summary
from per_alg import PrioritizedDoubleDQN, PrioritizedDQN
from proportional_per import ProportionalPER
from pseudoslam.envs.test import MonitorEnv
# from utils import get_player

MEMORY_SIZE = 2e5
MEMORY_WARMUP_SIZE = MEMORY_SIZE
IMAGE_SIZE = (64, 64)
CONTEXT_LEN = 4
FRAME_SKIP = 4
UPDATE_FREQ = 4
GAMMA = 0.99
LEARNING_RATE = 0.00025 / 4



def beta_adder(init_beta, step_size=0.0001):
    beta = init_beta
    step_size = step_size

    def adder():
        nonlocal beta, step_size
        beta += step_size
        return min(beta, 1)

    return adder


def process_transitions(transitions):
    transitions = np.array(transitions)
    batch_obs = np.stack(transitions[:, 0].copy())
    batch_act = transitions[:, 1].copy()
    batch_reward = transitions[:, 2].copy()
    batch_next_obs = np.expand_dims(np.stack(transitions[:, 3]), axis=1)
    batch_next_obs = np.concatenate([batch_obs, batch_next_obs],
                                    axis=1)[:, 1:, :, :].copy()
    batch_terminal = transitions[:, 4].copy()
    batch = (batch_obs, batch_act, batch_reward, batch_next_obs,
             batch_terminal)
    return batch


def run_episode(env, agent, per, mem=None, warmup=False, train=False):
    total_info = {}
    total_reward = 0
    all_cost = []
    traj = deque(maxlen=CONTEXT_LEN)
    obs = env.reset()
    for _ in range(CONTEXT_LEN - 1):
        traj.append(np.zeros(obs.shape))
    steps = 0
    if warmup:
        decay_exploration = False
    else:
        decay_exploration = True
    while True:
        steps += 1
        traj.append(obs)
        context = np.stack(traj, axis=0)
        action = agent.sample(context, decay_exploration=decay_exploration)
        next_obs, reward, terminal, info = env.step(action)
        rewards = reward
        for key in info.keys():
            if key not in total_info.keys():
                total_info[key] = info[key]
            else:
                total_info[key] += info[key]
            rewards +=info[key]
        transition = [obs, action, args.Rp*rewards, next_obs, terminal]
        if warmup:
            mem.append(transition)
        if train:
            per.store(transition)
            if steps % UPDATE_FREQ == 0:
                beta = get_beta()
                transitions, idxs, sample_weights = per.sample(beta=beta)
                batch = process_transitions(transitions)

                cost, delta = agent.learn(*batch, sample_weights)
                all_cost.append(cost)
                per.update(idxs, delta)

        total_reward += reward
        obs = next_obs
        if terminal:
            break

    return total_reward, steps, np.mean(all_cost),total_info


def run_evaluate_episode(env, agent):
    obs = env.reset()
    traj = deque(maxlen=CONTEXT_LEN)
    for _ in range(CONTEXT_LEN - 1):
        traj.append(np.zeros(obs.shape))
    total_reward = 0
    while True:
        traj.append(obs)
        context = np.stack(traj, axis=0)
        action = agent.predict(context)
        obs, reward, isOver, info = env.step(action)
        total_reward += reward
        if isOver:
            break
    return total_reward


def main():
    # Prepare environments
    # env = get_player(
    #     args.rom, image_size=IMAGE_SIZE, train=True, frame_skip=FRAME_SKIP)
    # test_env = get_player(
    #     args.rom,
    #     image_size=IMAGE_SIZE,
    #     frame_skip=FRAME_SKIP,
    #     context_len=CONTEXT_LEN)
    env = gym.make("pseudoslam:RobotExploration-v0")
    env = MonitorEnv(env,param={'goal':args.goal})

    # obs = env.reset()
    # print(obs.shape)
    # raise NotImplementedError
    # Init Prioritized Replay Memory
    per = ProportionalPER(alpha=0.6, seg_num=args.batch_size, size=MEMORY_SIZE)
    suffix = args.suffix+"_Rp{}_Goal{}".format(args.Rp,args.goal)
    logdir = os.path.join(args.logdir,suffix)
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    logger.set_dir(logdir)
    modeldir = os.path.join(args.modeldir,suffix)
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    
    # Prepare PARL agent
    act_dim = env.action_space.n
    model = AtariModel(act_dim)
    if args.alg == 'ddqn':
        algorithm = PrioritizedDoubleDQN(
            model, act_dim=act_dim, gamma=GAMMA, lr=LEARNING_RATE)
    elif args.alg == 'dqn':
        algorithm = PrioritizedDQN(
            model, act_dim=act_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = AtariAgent(algorithm, act_dim=act_dim, update_freq=UPDATE_FREQ)
    if os.path.exists(args.load):
        agent.restore(args.load)
    # Replay memory warmup
    total_step = 0
    with tqdm(total=MEMORY_SIZE, desc='[Replay Memory Warm Up]') as pbar:
        mem = []
        while total_step < MEMORY_WARMUP_SIZE:
            total_reward, steps, _ ,_= run_episode(
                env, agent, per, mem=mem, warmup=True)
            total_step += steps
            pbar.update(steps)
    per.elements.from_list(mem[:int(MEMORY_WARMUP_SIZE)])

    # env_name = args.rom.split('/')[-1].split('.')[0]
    test_flag = 0
    total_steps = 0
    pbar = tqdm(total=args.train_total_steps)
    save_steps =0 
    while total_steps < args.train_total_steps:
        # start epoch
        total_reward, steps, loss,info = run_episode(env, agent, per, train=True)
        total_steps += steps
        save_steps +=steps
        pbar.set_description('[train]exploration:{}'.format(agent.exploration))
        summary.add_scalar('train/score', total_reward,
                           total_steps)
        summary.add_scalar('train/loss', loss,
                           total_steps)  # mean of total loss
        summary.add_scalar('train/exploration',
                           agent.exploration, total_steps)
        summary.add_scalar('train/steps',
                           steps, total_steps)
        for key in info.keys():
            summary.add_scalar('train/'+key,
                           info[key], total_steps)
        pbar.update(steps)

        if total_steps // args.test_every_steps >= test_flag:
            print('start test!')
            while total_steps // args.test_every_steps >= test_flag:
                test_flag += 1
            pbar.write("testing")
            test_rewards = []
            for _ in tqdm(range(3), desc='eval agent'):
                eval_reward = run_evaluate_episode(env, agent)
                test_rewards.append(eval_reward)
            eval_reward = np.mean(test_rewards)
            logger.info(
                "eval_agent done, (steps, eval_reward): ({}, {})".format(
                    total_steps, eval_reward))
            summary.add_scalar('eval/reward', eval_reward,
                               total_steps)
        if save_steps>=100000:
            modeldir_ = os.path.join(modeldir,'itr_{}'.format(total_steps))
            if not os.path.exists(modeldir_):
                os.mkdir(modeldir_)
            print('save model!',modeldir_)
            agent.save(modeldir_)
            save_steps = 0

    pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument(
        '--alg',
        type=str,
        default="ddqn",
        help='dqn or ddqn, training algorithm to use.')
    parser.add_argument(
        '--train_total_steps',
        type=int,
        default=int(1e7),
        help='maximum environmental steps of games')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=100000,
        help='the step interval between two consecutive evaluations')
    parser.add_argument(
        '--logdir',
        type=str,
        default='/home/disk1/stone/paddlelog',
        help='logdir')
    parser.add_argument(
        '--modeldir',
        type=str,
        default='/home/disk1/stone/paddlemodel/',
        help='modeldir')
    parser.add_argument(
        '--suffix',
        type=str,
        default='first trian',
        help='suffix')
    parser.add_argument(
        '--Rp',
        type=float,
        default=1,
        help='Reward propotion')
    parser.add_argument(
        '--load',
        type=str,
        default='a',
        help='Load path')
    parser.add_argument(
        '--goal',
        type=float,
        default=0,
        help='Goal Reward')
    args = parser.parse_args()
    assert args.alg in ['dqn','ddqn'], \
        'used algorithm should be dqn or ddqn (double dqn)'
    get_beta = beta_adder(init_beta=0.5)
    main()
