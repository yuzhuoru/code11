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

from utils.delay_distribution import DoubleGaussianDistribution, GamaDistribution, UniformDistribution

class ActDelayWrapper(gym.Wrapper):
    """
    Wrapper for any non-RTRL environment, modelling random observation and action delays
    NB: alpha refers to the abservation delay, it is >= 0
    NB: The state-space now contains two different action delays:
        kappa is such that alpha+kappa is the index of the first action that was going to be applied when the observation started being captured, it is useful for the model
            (when kappa==0, it means that the delay is actually 1)
        beta is such that alpha+beta is the index of the last action that is known to have influenced the observation, it is useful for credit assignment (e.g. AC/DC)
            (alpha+beta is often 1 step bigger than the action buffer, and it is always >= 1)
    Kwargs:
        obs_delay_range: range in which alpha is sampled
        act_delay_range: range in which kappa is sampled
        initial_action: action (default None): action with which the action buffer is filled at reset() (if None, sampled in the action space)
    """

    def __init__(self, env, act_delay="gama", initial_action=None, skip_initial_actions=False):
        super().__init__(env)
        if act_delay == "gama":
            self.act_delay_dis = GamaDistribution()
        elif act_delay == "uniform":
            self.act_delay_dis = UniformDistribution()
        elif act_delay =="DoubleGaussian":
            self.act_delay_dis = DoubleGaussianDistribution()

        self.wrapped_env = env
        self.min_action_delay = 1
        self.sup_action_delay = self.act_delay_dis.max_delay + 1
        
        '''
        self.observation_space = spaces.Dict({
            "observation": env.observation_space, # most recent observation
            "predict_observation":env.observation_space,  # predict observation
            "action_delay":spaces.Discrete(sup_action_delay+1)  # action delay
        })
        '''

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
        self.t = - (self.sup_action_delay)  # this is <= -2
        while self.t < 0:
            act = self.action_space.sample() if self.initial_action is None else self.initial_action
            delay = self.send_action(act, init=True)  # TODO : initialize this better
            self.t += 1
        # self.receive_action()  # an action has to be applied
        self.predict_state = None
        self.last_obs = first_observation
        assert self.t == 0

        # print("DEBUG: end of reset ---")
        # print(f"DEBUG: self.past_actions:{self.past_actions}")
        # print(f"DEBUG: self.past_observations:{self.past_observations}")
        # print(f"DEBUG: self.arrival_times_actions:{self.arrival_times_actions}")
        # print(f"DEBUG: self.arrival_times_observations:{self.arrival_times_observations}")
        # print(f"DEBUG: self.t:{self.t}")
        # print("DEBUG: ---")
        return first_observation, {}

    def step(self, action):

        """
        When kappa is 0 and alpha is 0, this is equivalent to the RTRL setting
        (The inference time is NOT considered part of beta or kappa)
        """

        # at the brain
        delay = self.send_action(action) #返回当前delay
        # at the remote actor
        _, _ = self.receive_action()  #返回接收到动作的delay
        
        m, r, terminated, truncated, info = self.env.step(self.next_action)  # before receive_action (e.g. rtrl setting with 0 delays)
        self.last_obs = m
        self.t += 1

        # print("DEBUG: end of step ---")
        # print(f"DEBUG: self.past_actions:{self.past_actions}")
        # print(f"DEBUG: self.past_observations:{self.past_observations}")
        # print(f"DEBUG: self.arrival_times_actions:{self.arrival_times_actions}")
        # print(f"DEBUG: self.arrival_times_observations:{self.arrival_times_observations}")
        # print(f"DEBUG: self.t:{self.t}")
        # print("DEBUG: ---")
        
        return m, r, terminated, truncated, info

    def send_action(self, action, init=False):
        """
        Appends action to the left of self.past_actions
        Simulates the time at which it will reach the agent and stores it on the left of self.arrival_times_actions
        """
        # at the brain
        trueDelay = self.act_delay_dis.dis_sample()
        delay = trueDelay if not init else 0  # TODO: change this if we implement a different initialization
        self.arrival_times_actions.appendleft(self.t + trueDelay)
        
        self.past_actions.appendleft(action)
        return delay

    def receive_action(self):
        """
        Looks for the last created action that has arrived before t at the agent
        NB: since it is the most recently created action that the agent got, this is the one that is to be applied
        Returns:
            next_action_idx: int: the index of the action that is going to be applied
            prev_action_idx: int: the index of the action previously being applied (i.e. of the action that influenced the observation since it is retrieved instantaneously in usual Gym envs)
        """
        # CAUTION: from the brain point of view, the "previous action"'s age (kappa_t) is not like the previous "next action"'s age (beta_{t-1}) (e.g. repeated observations)
        prev_action_idx = self.prev_action_idx + 1  # + 1 is to account for the fact that this was the right idx 1 time-step before
        next_action_idx = next(i for i, t in enumerate(self.arrival_times_actions) if t <= self.t)
        self.prev_action_idx = next_action_idx
        self.next_action = self.past_actions[next_action_idx]
        
        # print(f"DEBUG: next_action_idx:{next_action_idx}, prev_action_idx:{prev_action_idx}")
        return next_action_idx, prev_action_idx

