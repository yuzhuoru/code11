import numpy as np
import sys
import torch
from utils.common_utils import set_seed
from utils.delay_distribution import DoubleGaussianDistribution, GamaDistribution, UniformDistribution

__all__ = ["ReplayBuffer_delayaware"]


def combined_shape(length: int, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class ReplayBufferDelayaware:
    """
    Implementation of replay buffer with uniform sampling probability.
    """

    def __init__(self, index=0, **kwargs):
       
        set_seed(kwargs["trainer"], kwargs["seed"], index + 100)
        self.obsv_dim = kwargs["obsv_dim"]
        self.act_dim = kwargs["action_dim"]
        self.max_size = kwargs["buffer_max_size"]
        self.buf = {
            "obs": np.zeros(
                combined_shape(self.max_size, self.obsv_dim), dtype=np.float32
            ),
            "obs2": np.zeros(
                combined_shape(self.max_size, self.obsv_dim), dtype=np.float32
            ),
            "act": np.zeros(
                combined_shape(self.max_size, self.act_dim), dtype=np.float32
            ),
            "rew": np.zeros(self.max_size, dtype=np.float32),
            "done": np.zeros(self.max_size, dtype=np.float32),
            "logp": np.zeros(self.max_size, dtype=np.float32),
        }
        self.additional_info = kwargs["additional_info"]
        for k, v in self.additional_info.items():
            self.buf[k] = np.zeros(
                combined_shape(self.max_size, v["shape"]), dtype=v["dtype"]
            )
            self.buf["next_" + k] = np.zeros(
                combined_shape(self.max_size, v["shape"]), dtype=v["dtype"]
            )
        self.ptr, self.size, = (
            0,
            0,
        )
        
        #total max delay 
        if kwargs["delay_mode"] == "act":
            if kwargs["act_delay_dis"] == "gama":
                act_delay_dis = GamaDistribution()
            elif kwargs["act_delay_dis"] == "uniform":
                act_delay_dis = UniformDistribution()
            elif kwargs["act_delay_dis"] == "DoubleGaussian":
                act_delay_dis = DoubleGaussianDistribution()
            self.delay_max = act_delay_dis.max_delay
            
        elif kwargs["delay_mode"] == "obs":
            if kwargs["obs_delay_dis"] == "gama":
                obs_delay_dis = GamaDistribution()
            elif kwargs["obs_delay_dis"] == "uniform":
                obs_delay_dis = UniformDistribution()
            elif kwargs["obs_delay_dis"] == "DoubleGaussian":
                obs_delay_dis = DoubleGaussianDistribution()
            self.delay_max = obs_delay_dis.max_delay
            
        elif kwargs["delay_mode"] == "both":
            if kwargs["obs_delay_dis"] == "gama":
                obs_delay_dis = GamaDistribution()
            elif kwargs["obs_delay_dis"] == "uniform":
                obs_delay_dis = UniformDistribution()
            elif kwargs["obs_delay_dis"] == "DoubleGaussian":
                obs_delay_dis = DoubleGaussianDistribution()

            if kwargs["act_delay_dis"] == "gama":
                act_delay_dis = GamaDistribution()
            elif kwargs["act_delay_dis"] == "uniform":
                act_delay_dis = UniformDistribution()
            elif kwargs["act_delay_dis"] == "DoubleGaussian":
                act_delay_dis = DoubleGaussianDistribution()
            self.delay_max = obs_delay_dis.max_delay+act_delay_dis.max_delay
            

    def __len__(self):
        return self.size

    def __get_RAM__(self):
        return int(sys.getsizeof(self.buf)) * self.size / (self.max_size * 1000000)

    def store(
        self,
        obs: np.ndarray,
        info: dict,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
        logp: np.ndarray,
        next_info: dict,
    ):
        self.buf["obs"][self.ptr] = obs
        self.buf["obs2"][self.ptr] = next_obs
        self.buf["act"][self.ptr] = act
        self.buf["rew"][self.ptr] = rew
        self.buf["done"][self.ptr] = done
        self.buf["logp"][self.ptr] = logp
        for k in self.additional_info.keys():
            self.buf[k][self.ptr] = info[k]
            self.buf["next_" + k][self.ptr] = next_info[k]
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, samples: list):
        for sample in samples:
            self.store(*sample)

    def sample_batch(self, batch_size: int):
        idxs = np.random.randint(0, self.size - self.delay_max, size=batch_size)
        batch = {}
        for k, v in self.buf.items():
            if len(v.shape) == 2:
                shape = (self.delay_max+1,batch_size,v.shape[1])
            else:
                shape = (self.delay_max+1,batch_size)

            _batch = np.zeros(shape)
            for i in range(0,self.delay_max+1,1):
                _batch[i] = v[idxs+i]
            batch[k] = _batch
        
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}