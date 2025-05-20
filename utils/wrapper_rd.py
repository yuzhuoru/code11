from collections import deque
from random import sample
import itertools

import gymnasium as gym
from gymnasium.spaces import Tuple, Discrete

import numpy as np

from utils.delay_distribution import DoubleGaussianDistribution, GammaDistribution, UniformDistribution


class ObsAndActDelayWrapper(gym.Wrapper):
    def __init__(self, env, act_delay="gamma", obs_delay="gamma", initial_action=None, skip_initial_actions=False):
        super().__init__(env)
        if act_delay == "gamma":
            self.act_delay_dis = GammaDistribution()
        elif act_delay == "uniform":
            self.act_delay_dis = UniformDistribution()
        elif act_delay =="doublegaussian":
            self.act_delay_dis = DoubleGaussianDistribution()

        if obs_delay == "gamma":
            self.obs_delay_dis = GammaDistribution()
        elif obs_delay == "uniform":
            self.obs_delay_dis = UniformDistribution()
        elif obs_delay =="doublegaussian":
            self.obs_delay_dis = DoubleGaussianDistribution()

        self.wrapped_env = env
        self.initial_action = initial_action
        self.skip_initial_actions = skip_initial_actions
        self.past_actions = deque(maxlen=(self.obs_delay_dis.max_delay + 1) + (self.act_delay_dis.max_delay + 1))
        self.past_observations = deque(maxlen=self.obs_delay_dis.max_delay + 1)
        self.arrival_times_actions = deque(maxlen=self.act_delay_dis.max_delay + 1)
        self.arrival_times_observations = deque(maxlen=self.obs_delay_dis.max_delay + 1)

        self.t = 0
        self.done_signal_sent = False
        self.next_action = None
        self.cum_rew_actor = 0.
        self.cum_rew_brain = 0.
        self.prev_action_idx = 0  # TODO : initialize this better

    def reset(self, **kwargs):
        self.cum_rew_actor = 0.
        self.cum_rew_brain = 0.
        self.prev_action_idx = 0  # TODO : initialize this better
        self.done_signal_sent = False
        first_observation, _ = super().reset(**kwargs)

        # fill up buffers
        self.t = - ((self.obs_delay_dis.max_delay + 1) + (self.act_delay_dis.max_delay + 1))  
        while self.t < 0:
            act = self.action_space.sample() if self.initial_action is None else self.initial_action
            self.send_action(act, init=True)  # TODO : initialize this better
            self.send_observation((first_observation, 0., False, False, {}, 0, 1))  # TODO : initialize this better
            self.t += 1
        self.receive_action()  # an action has to be applied

        assert self.t == 0
        received_observation, *_ = self.receive_observation()
        return received_observation, {}

    def step(self, action):
        self.send_action(action)
        true_obs = action
        if self.t < self.act_delay_dis.max_delay+1 and self.skip_initial_actions:
            self.receive_action()
        elif self.done_signal_sent:
            self.send_observation(self.past_observations[0])
        else:
            m, r, terminated, truncated, info = self.env.step(self.next_action)  
            true_obs = m
            kappa, beta = self.receive_action()
            self.cum_rew_actor += r
            self.done_signal_sent = terminated or truncated
            self.send_observation((m, self.cum_rew_actor, terminated, truncated, info, kappa, beta))

        m, cum_rew_actor_delayed, terminated, truncated, info = self.receive_observation()
        r = cum_rew_actor_delayed - self.cum_rew_brain
        self.cum_rew_brain = cum_rew_actor_delayed
        
        self.t += 1
        return m, r, terminated, truncated, info

    def send_action(self, action, init=False):
        kappa = self.act_delay_dis.dis_sample() if not init else 0  
        self.arrival_times_actions.appendleft(self.t + kappa)
        self.past_actions.appendleft(action)

    def receive_action(self):
        prev_action_idx = self.prev_action_idx + 1  
        next_action_idx = next(i for i, t in enumerate(self.arrival_times_actions) if t <= self.t)
        self.prev_action_idx = next_action_idx
        self.next_action = self.past_actions[next_action_idx]
        return next_action_idx, prev_action_idx

    def send_observation(self, obs):
        alpha = self.obs_delay_dis.dis_sample()
        self.arrival_times_observations.appendleft(self.t + alpha)
        self.past_observations.appendleft(obs)

    def receive_observation(self):
        alpha = next(i for i, t in enumerate(self.arrival_times_observations) if t <= self.t)
        m, r, terminated, truncated, info, kappa, beta = self.past_observations[alpha]
        return m, r, terminated, truncated, info