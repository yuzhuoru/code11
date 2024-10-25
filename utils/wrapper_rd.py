from collections import deque
from random import sample
import itertools

import gymnasium as gym
from gymnasium.spaces import Tuple, Discrete

import numpy as np

from utils.delay_distribution import DoubleGaussianDistribution, GamaDistribution, UniformDistribution


class ObsAndActDelayWrapper(gym.Wrapper):
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

    def __init__(self, env, act_delay="gama", obs_delay="gama", initial_action=None, skip_initial_actions=False):
        super().__init__(env)

        if act_delay == "gama":
            self.act_delay_dis = GamaDistribution()
        elif act_delay == "uniform":
            self.act_delay_dis = UniformDistribution()
        elif act_delay =="DoubleGaussian":
            self.act_delay_dis = DoubleGaussianDistribution()

        if obs_delay == "gama":
            self.obs_delay_dis = GamaDistribution()
        elif obs_delay == "uniform":
            self.obs_delay_dis = UniformDistribution()
        elif obs_delay =="DoubleGaussian":
            self.obs_delay_dis = DoubleGaussianDistribution()

        self.wrapped_env = env

        '''        
        self.observation_space = Tuple((
            env.observation_space,  # most recent observation
            Tuple([env.action_space] * (obs_delay_range.stop + act_delay_range.stop - 1)),  # action buffer
            Discrete(obs_delay_range.stop),  # observation delay int64
            Discrete(act_delay_range.stop),  # action delay int64
        ))
        '''

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
        self.t = - ((self.obs_delay_dis.max_delay + 1) + (self.act_delay_dis.max_delay + 1))  # this is <= -2
        while self.t < 0:
            act = self.action_space.sample() if self.initial_action is None else self.initial_action
            self.send_action(act, init=True)  # TODO : initialize this better
            self.send_observation((first_observation, 0., False, False, {}, 0, 1))  # TODO : initialize this better
            self.t += 1
        self.receive_action()  # an action has to be applied

        assert self.t == 0
        received_observation, *_ = self.receive_observation()
        # print("DEBUG: end of reset ---")
        # print(f"DEBUG: self.past_actions:{self.past_actions}")
        # print(f"DEBUG: self.past_observations:{self.past_observations}")
        # print(f"DEBUG: self.arrival_times_actions:{self.arrival_times_actions}")
        # print(f"DEBUG: self.arrival_times_observations:{self.arrival_times_observations}")
        # print(f"DEBUG: self.t:{self.t}")
        # print("DEBUG: ---")
        return received_observation, {}

    def step(self, action):
        """
        When kappa is 0 and alpha is 0, this is equivalent to the RTRL setting
        (The inference time is NOT considered part of beta or kappa)
        """
        # at the brain
        self.send_action(action)
        true_obs = action
        # at the remote actor
        if self.t < self.act_delay_dis.max_delay+1 and self.skip_initial_actions:
            # assert False, "skip_initial_actions==True is not supported"
            # do nothing until the brain's first actions arrive at the remote actor
            self.receive_action()
        elif self.done_signal_sent:
            # just resend the last observation until the brain gets it
            self.send_observation(self.past_observations[0])
        else:
            m, r, terminated, truncated, info = self.env.step(self.next_action)  # before receive_action (e.g. rtrl setting with 0 delays)
            true_obs = m
            kappa, beta = self.receive_action()
            self.cum_rew_actor += r
            self.done_signal_sent = terminated or truncated
            self.send_observation((m, self.cum_rew_actor, terminated, truncated, info, kappa, beta))

        # at the brain again
        m, cum_rew_actor_delayed, terminated, truncated, info = self.receive_observation()
        r = cum_rew_actor_delayed - self.cum_rew_brain
        self.cum_rew_brain = cum_rew_actor_delayed

        '''
        print("timestep:",self.t,"真实obs:", true_obs, " 收到obs:",m)
        print("obs queue:",self.past_observations)
        '''
        
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
        kappa = self.act_delay_dis.dis_sample() if not init else 0  # TODO: change this if we implement a different initialization
        self.arrival_times_actions.appendleft(self.t + kappa)
        self.past_actions.appendleft(action)

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

    def send_observation(self, obs):
        """
        Appends obs to the left of self.past_observations
        Simulates the time at which it will reach the brain and appends it in self.arrival_times_observations
        """
        # at the remote actor
        alpha = self.obs_delay_dis.dis_sample()
        self.arrival_times_observations.appendleft(self.t + alpha)
        self.past_observations.appendleft(obs)

    def receive_observation(self):
        """
        Looks for the last created observation at the agent/observer that reached the brain at time t
        NB: since this is the most recently created observation that the brain got, this is the one currently being considered as the last observation
        Returns:
            augmented_obs: tuple:
                m: object: last observation that reached the brain
                past_actions: tuple: the history of actions that the brain sent so far
                alpha: int: number of micro time steps it took the last observation to travel from the agent/observer to the brain
                kappa: int: action travel delay + number of micro time-steps for which the next action has been applied at the agent
                beta: int: action travel delay + number of micro time-steps for which the previous action has been applied at the agent
            r: float: delayed reward corresponding to the transition that created m
            d: bool: delayed done corresponding to the transition that created m
            info: dict: delayed info corresponding to the transition that created m
        """
        # at the brain
        alpha = next(i for i, t in enumerate(self.arrival_times_observations) if t <= self.t)
        
        m, r, terminated, truncated, info, kappa, beta = self.past_observations[alpha]
        return m, r, terminated, truncated, info