import functools
import gc
import math
import random
import string
from typing import List, Optional, Tuple, Callable, Union

import numpy as np
import torch
from torch.backends import cudnn, opt_einsum
from torch.utils._pytree import tree_map

from torch.distributed.tensor import distribute_tensor, DTensor

def to_dist(x, from_local=False, **meta):
    if from_local:
        return DTensor.from_local(
            x,
            device_mesh=meta["device_mesh"],
            placements=meta["placements"],
            shape=meta["shape"],
            stride=meta["stride"],
        )
    else:
        return distribute_tensor(x, device_mesh=meta["device_mesh"], placements=meta["placements"])


def to_local(x, keep_sharded=False):
    if isinstance(x, DTensor):
        meta = dict(
            device_mesh=x.device_mesh,
            placements=x.placements,
            shape=x.shape,
            stride=x.stride(),
        )
        if keep_sharded:
            return x.to_local(), meta
        else:
            return x.full_tensor(), meta

    return x, None


def local_op(x, fn, keep_sharded=False):
    """
    converts to Tensor, does a thing, then back to Dtensor
    """
    x, meta = to_local(x, keep_sharded)
    x = fn(x)
    if meta is not None:
        x = to_dist(x, from_local=keep_sharded, **meta)
    return x