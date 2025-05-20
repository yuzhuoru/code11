from collections import deque
import math
from random import sample
import itertools
from typing import Optional
import gymnasium as gym
from gymnasium.spaces import Dict, Tuple, Discrete, Box
from gymnasium import spaces, logger
import numpy as np
import copy
import torch

from utils.delay_distribution import DoubleGaussianDistribution, GammaDistribution, UniformDistribution

class ActDelayWrapper(gym.Wrapper):

    def __init__(self, env, act_delay="gamma", initial_action=None, skip_initial_actions=False):
        super().__init__(env)
        if act_delay == "gamma":
            self.act_delay_dis = GammaDistribution()
        elif act_delay == "uniform":
            self.act_delay_dis = UniformDistribution()
        elif act_delay =="doublegaussian":
            self.act_delay_dis = DoubleGaussianDistribution()

        self.wrapped_env = env
        self.min_action_delay = 1
        self.sup_action_delay = self.act_delay_dis.max_delay + 1

        self.initial_action = initial_action
        self.skip_initial_actions = skip_initial_actions
        self.past_actions = deque(maxlen=self.sup_action_delay)
        self.arrival_times_actions = deque(maxlen=self.sup_action_delay)

        self.t = 0
        self.done_signal_sent = False
        self.next_action = None
        self.cum_rew_actor = 0.
        self.cum_rew_brain = 0.
        self.prev_action_idx = 0  # TODO : initialize this better

        self.predict_state = None
        self.last_obs = None
        

    def reset(self,seed=None, options=None):

        self.cum_rew_actor = 0.
        self.cum_rew_brain = 0.
        self.prev_action_idx = 0  # TODO : initialize this better
        first_observation, _ = super().reset()
        # fill up buffers
        self.t = - (self.sup_action_delay) 
        while self.t < 0:
            act = self.action_space.sample() if self.initial_action is None else self.initial_action
            delay = self.send_action(act, init=True)  # TODO : initialize this better
            self.t += 1
        self.predict_state = None
        self.last_obs = first_observation
        assert self.t == 0

        return first_observation, {}

    def step(self, action):

        # at the brain
        delay = self.send_action(action) 
        # at the remote actor
        _, _ = self.receive_action() 
        
        m, r, terminated, truncated, info = self.env.step(self.next_action) 
        self.last_obs = m
        self.t += 1
        
        return m, r, terminated, truncated, info

    def send_action(self, action, init=False):
        # at the brain
        trueDelay = self.act_delay_dis.dis_sample()
        delay = trueDelay if not init else 0  
        self.arrival_times_actions.appendleft(self.t + trueDelay)
        
        self.past_actions.appendleft(action)
        return delay

    def receive_action(self):
        prev_action_idx = self.prev_action_idx + 1  # + 1 is to account for the fact that this was the right idx 1 time-step before
        next_action_idx = next(i for i, t in enumerate(self.arrival_times_actions) if t <= self.t)
        self.prev_action_idx = next_action_idx
        self.next_action = self.past_actions[next_action_idx]
        return next_action_idx, prev_action_idx

