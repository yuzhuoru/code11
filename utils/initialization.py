import importlib
import os.path
import sys

import gymnasium as gym
from utils.wrapper_ad import ActDelayWrapper
from utils.wrapper_rd import ObsAndActDelayWrapper
from wrapping_env import wrapping_env
from utils.wrapper_od import ObsDelayWrapper

def create_env(**kwargs):
    env_name = kwargs["env_id"]
    env = gym.make(env_name)

    # Wrapping the env
    max_episode_steps = kwargs.get("max_episode_steps", None)
    reward_scale = kwargs.get("reward_scale", None)
    reward_shift = kwargs.get("reward_shift", None)

    env = wrapping_env(
        env=env,
        max_episode_steps=max_episode_steps,
        reward_shift=reward_shift,
        reward_scale=reward_scale,
    )

    # delay env wrapper
    if kwargs["delay_mode"] == "act":
        print("!!! act delay mode !!!",kwargs["act_delay_dis"])
        env = ActDelayWrapper(env, act_delay=kwargs["act_delay_dis"])
    elif kwargs["delay_mode"] == "obs":
        print("!!! obs delay mode !!!",kwargs["obs_delay_dis"])
        env = ObsDelayWrapper(env, obs_delay=kwargs["obs_delay_dis"])
    elif kwargs["delay_mode"] == "both":
        print("!!! both obs and act delay mode !!!",kwargs["act_delay_dis"],kwargs["obs_delay_dis"])
        env = ObsAndActDelayWrapper(env, act_delay=kwargs["act_delay_dis"], obs_delay=kwargs["obs_delay_dis"])
    
    print("Create environment successfully!")
    return env


def create_alg(**kwargs):
    alg_name = kwargs["algorithm"]
    alg_file_name = alg_name.lower()
    try:
        module = importlib.import_module(alg_file_name)
    except NotImplementedError:
        raise NotImplementedError("This algorithm does not exist")

    if hasattr(module, alg_name):
        alg_cls = getattr(module, alg_name)
        alg = alg_cls(**kwargs)
    else:
        raise NotImplementedError("This algorithm is not properly defined")

    print("Create algorithm successfully!")
    return alg


def create_apprfunc(**kwargs):
    apprfunc_name = kwargs["apprfunc"]
    apprfunc_file_name = apprfunc_name.lower()
    try:
        file = importlib.import_module('networks.' + apprfunc_file_name)
    except NotImplementedError:
        raise NotImplementedError("This apprfunc does not exist")

    # name = kwargs['name'].upper()

    name = formatter(kwargs["name"])
    # print(name)
    # print(kwargs)

    if hasattr(file, name):  #
        apprfunc_cls = getattr(file, name)
        apprfunc = apprfunc_cls(**kwargs)
    else:
        raise NotImplementedError("This apprfunc is not properly defined")

    # print("--Initialize appr func: " + name + "...")
    return apprfunc


def create_buffer(**kwargs):
    buffer_file_name = kwargs["buffer_name"].lower()
    try:
        module = importlib.import_module("training." + buffer_file_name)
    except NotImplementedError:
        raise NotImplementedError("This buffer does not exist")

    buffer_name = formatter(buffer_file_name)

    if hasattr(module, buffer_name):  #
        buffer_cls = getattr(module, buffer_name)  #
        buffer = buffer_cls(**kwargs)
    else:
        raise NotImplementedError("This buffer is not properly defined")

    print("Create buffer successfully!")
    return buffer


def formatter(src: str, firstUpper: bool = True):
    arr = src.split("_")
    res = ""
    for i in arr:
        res = res + i[0].upper() + i[1:]

    if not firstUpper:
        res = res[0].lower() + res[1:]
    return res
