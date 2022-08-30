import os
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import contextmanager

from enum import Enum, auto



class DotDict(dict):
    """dot.notation access to dictionary attributes
        See https://stackoverflow.com/questions/49901590/python-using-copy-deepcopy-on-dotdict
    """

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



class DataParamMode(Enum): # TODO clean
    INSTANCE_PARAMS = auto()
    DISABLED = auto()



class LabelDisturbanceMode(Enum):
    FLIP_ROLL = auto()
    AFFINE = auto()



def in_notebook():
    try:
        get_ipython().__class__.__name__
        return True
    except NameError:
        return False



def get_script_dir(_fll):
    if in_notebook:
        return os.path.abspath('')
    else:
        return os.path.dirname(os.path.realpath(_fll))
