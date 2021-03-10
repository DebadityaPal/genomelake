from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import shutil

import numpy as np
import hub

def write_hub(arr, path, overwrite=True):
    """Write a hub dataset to disk
    """
    if os.path.exists(path) and os.path.isdir(path) and overwrite:
        shutil.rmtree(path)

    if os.path.exists(path):
        raise FileExistsError("Output path {} already exists".format(path))

    if arr.ndim == 1:
        schema = {
            "value" : hub.schema.Tensor(arr.shape[0])
        }
        dataset = hub.Dataset(path, shape=(1,), schema=schema, mode='w')
        dataset["value", 0][:] = arr.astype(np.float32)
        dataset.flush()
        dataset.close()

    elif arr.ndim == 2:
        schema = {
            "value" : hub.schema.Tensor(arr.shape[1])
        }
        dataset = hub.Dataset(path, shape=(arr.shape[0],), schema=schema, mode='w')
        dataset["value"][:] = arr.astype(np.float32)
        dataset.flush()
        dataset.close()
    else:
        raise ValueError("hub backend only supports 1D or 2D arrays")

def load_hub(path):
    return HubDataset(path)

class HubDataset(object):
    """A read-only wrapper of hub.Dataset"""

    def __init__(self, array):
        if isinstance(array, hub.api.dataset.Dataset):
            self._arr = array
        else:
            self._arr = hub.Dataset(array, mode="r")

    def __getitem__(self, key):
        return self._arr[key]["value"].compute()

    def __setitem__(self, key, item):
        raise NotImplemented("TDBDenseArray is read-only")

    @property
    def shape(self):
        return self._arr.shape

    @property
    def ndim(self):
        if self._arr.shape == (1, ):
            return 1
        else:
            return 2
