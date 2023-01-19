import torch
import collections

class StageIterator(collections.abc.Iterator):
    def __init__(self, stages, verbose=False):
        super().__init__()
        self.stages = iter(stages)
        self.previous = None
        self.idx = -1
        self.len = len(stages)
        self.verbose = verbose

    def __next__(self):
        if self.previous is None:
            self.previous = next(self.stages)
        else:
            nxt = next(self.stages)
            for key, value in self.previous.items():
                if not key in nxt:
                    nxt[key] = value
            self.previous = nxt
        self.idx += 1
        if self.verbose:
            print(f"Opening stage {self.idx+1}/{self.len}")
        return self.previous



class Stage(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if '__activate_fn__' in kwargs:
            self.__activate_fn__ = kwargs['__activate_fn__']
            del kwargs['__activate_fn__']
        else:
            self.__activate_fn__ = lambda self, *args, **kwargs: None

    def set_activate_fn(self, lmbd):
        self.__activate_fn__ = lmbd

    def activate(self, *args, **kwargs):
        self.__activate_fn__(self, *args, **kwargs)



class RegularizationParam():
    def __init__(self, name, target_val, lambda_r, active, call_fn):
        self.name = name
        self.target_val = target_val
        self.lambda_r = lambda_r
        self.active = active
        self.call_fn = call_fn

    def set_call_fn(self, lmbd):
        self.call_fn = lmbd

    def __call__(self):

        if self.active:
            return self.lambda_r * self.call_fn(self.target_val)
        return torch.tensor(0.0)

    def __repr__(self):
        return f'{self.name} -> {self.target_val} (active={self.active}, lambda_r={self.lambda_r})'



def init_regularization_params(name_list, target_val=0.0, lambda_r=0.0, active=False):
    params = {}
    for name in name_list:
        params[name] = RegularizationParam(
            name, target_val, lambda_r, active, lambda: None
        )
    return params