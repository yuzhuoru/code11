from collections import deque
from random import sample
import itertools

import gymnasium as gym
from gymnasium.spaces import Tuple, Discrete

import numpy as np

from utils.delay_distribution import DoubleGaussianDistribution, GamaDistribution, UniformDistribution


class ObsDelayWrapper(gym.Wrapper):
    """
    Wrapper for any non-RTRL environment, modelling random observation delays
    NB: alpha refers to the observation delay, it is >= 0
    Kwargs:
        obs_delay_range: range in which alpha is sampled
        initial_action: action (default None): action with which the action buffer is filled at reset() (if None, sampled in the action space)
    """

    def __init__(self, env, obs_delay="gama", initial_action=None, skip_initial_actions=False):
        super().__init__(env)

        if obs_delay == "gama":
            self.obs_delay_dis = GamaDistribution()
        elif obs_delay == "uniform":
            self.obs_delay_dis = UniformDistribution()
        elif obs_delay =="DoubleGaussian":
            self.obs_delay_dis = DoubleGaussianDistribution()

        self.wrapped_env = env

        self.initial_action = initial_action
        self.skip_initial_actions = skip_initial_actions
        self.past_observations = deque(maxlen=self.obs_delay_dis.max_delay+1)
        self.arrival_times_observations = deque(maxlen=self.obs_delay_dis.max_delay+1)

        self.t = 0
        self.done_signal_sent = False
        self.cum_rew_actor = 0.
        self.cum_rew_brain = 0.

    def reset(self, **kwargs):
        self.cum_rew_actor = 0.
        self.cum_rew_brain = 0.
        self.done_signal_sent = False
        first_observation, _ = super().reset(**kwargs)

        # fill up buffers
        self.t = - (self.obs_delay_dis.max_delay+1)  # this is <= -1
        while self.t < 0:
            self.send_observation((first_observation, 0., False, False, {}, 0))
            self.t += 1

        assert self.t == 0
        received_observation, *_ = self.receive_observation()
        return received_observation, {}

    def step(self, action):
        """
        Handles environment step with observation delay.
        """

        # at the brain
        true_obs = action
        if self.done_signal_sent:
            # just resend the last observation until the brain gets it
            self.send_observation(self.past_observations[0])
        else:
            m, r, terminated, truncated, info = self.env.step(action)
            true_obs = m
            self.cum_rew_actor += r
            self.done_signal_sent = terminated or truncated
            self.send_observation((m, self.cum_rew_actor, terminated, truncated, info, 0))

        # at the brain again
        m, cum_rew_actor_delayed, terminated, truncated, info = self.receive_observation()
        r = cum_rew_actor_delayed - self.cum_rew_brain
        self.cum_rew_brain = cum_rew_actor_delayed
        '''
        print("timestep:",self.t,"真实obs:", true_obs, " 收到obs:",m)
        print("obs queue:",self.past_observations)
        print("obs arrival queue:",self.arrival_times_observations)
        '''
        self.t += 1

        return m, r, terminated, truncated, info

    def send_observation(self, obs):
        """
        Appends obs to the left of self.past_observations
        Simulates the time at which it will reach the brain and appends it in self.arrival_times_observations
        """
        alpha = self.obs_delay_dis.dis_sample()
        self.arrival_times_observations.appendleft(self.t + alpha)
        self.past_observations.appendleft(obs)
        '''
        print("------------send----------------")
        print("t:",self.t,"  alpha:",alpha,"  t+alpha:",self.t + alpha)
        print("arrval_obs_queue:",self.arrival_times_observations)
        print("past_obs_queue:",self.past_observations)
        '''

    def receive_observation(self):
        """
        Looks for the last created observation at the agent/observer that reached the brain at time t
        NB: since this is the most recently created observation that the brain got, this is the one currently being considered as the last observation
        Returns:
            augmented_obs: tuple:
                m: object: last observation that reached the brain
                alpha: int: number of micro time steps it took the last observation to travel from the agent/observer to the brain
        """
        alpha = next(i for i, t in enumerate(self.arrival_times_observations) if t <= self.t)
        
        m, r, terminated, truncated, info, _ = self.past_observations[alpha]
        '''
        print("------------receive----------------")
        print("t:",self.t)
        print("rec obs:",m)
        '''
        return m, r, terminated, truncated, info
