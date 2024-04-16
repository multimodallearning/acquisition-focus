import torch
import collections


class StageIterator(collections.abc.Iterator):
    def __init__(self, stages, verbose=False):
        super().__init__()
        self.stages = stages
        self.stage_keys = list(stages.keys())
        self.current = None
        self.idx = -1
        self.len = len(stages)
        self.verbose = verbose

    def __next__(self):
        if self.current is None:
            self.current_key = self.stage_keys.pop(0)
            self.current = self.stages[self.current_key]
        else:
            if not self.stage_keys: raise StopIteration()
            nxt_key = self.stage_keys.pop(0)
            nxt = self.stages[nxt_key]
            for key, value in self.current.items():
                if not key in nxt:
                    nxt[key] = value
            self.current_key = nxt_key
            self.current = nxt
        self.idx += 1

        if self.verbose:
            print(f"Opening stage '{self.current_key}' ({self.idx+1}/{self.len})")
        return self.current



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
