import torch
import collections
from slice_inflate.models.affine_transform import get_theta_params

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



def get_atm_angle_mean_closure(atm):
    def closure(target_val):
        theta_ap = get_theta_params(atm.last_theta)[0]
        return ((theta_ap-target_val.to(theta_ap))**2).mean()
    return closure



def get_atm_angle_std_closure(atm):
    def closure(target_val):
        theta_ap = get_theta_params(atm.last_theta)[0]
        std = theta_ap.std(0)
        std[std.isnan()] = 0.
        return (std-target_val.to(theta_ap)).abs()
    return closure



def get_atm_offset_mean_closure(atm):
    def closure(target_val):
        theta_tp = get_theta_params(atm.last_theta_t)[1]
        return ((theta_tp-target_val.to(theta_tp))**2).mean()
    return closure



def get_atm_offset_std_closure(atm):
    def closure(target_val):
        theta_tp = get_theta_params(atm.last_theta_t)[1]
        std = theta_tp.std(0)
        std[std.isnan()] = 0.
        return (std-target_val.to(std)).abs()

    return closure



def optimize_sa_angles(stage):
    if 'epoch_sa_angles_mean' in stage:
        stage['sa_atm'].set_init_theta_ap(stage['epoch_sa_angles_mean'])

    r_params = stage['r_params']
    r_params['sa_angles'].active = False
    r_params['sa_offsets'].active = True
    r_params['sa_offsets'].target_val = torch.zeros(3)
    r_params['sa_offsets'].lambda_r = 0.5

    hla_atm = stage['hla_atm']
    sa_atm = stage['sa_atm']
    r_params['hla_angles'].set_call_fn(get_atm_angle_mean_closure(hla_atm))
    r_params['hla_offsets'].set_call_fn(get_atm_offset_mean_closure(hla_atm))
    r_params['sa_angles'].set_call_fn(get_atm_angle_mean_closure(sa_atm))
    r_params['sa_offsets'].set_call_fn(get_atm_offset_mean_closure(sa_atm))


    print('Init sa_theta_ap', stage['sa_atm'].init_theta_ap)
    [print(name, rp) for name, rp in r_params.items()]



def optimize_hla_angles(stage):

    r_params = stage['r_params']
    r_params['sa_angles'].active = False
    r_params['sa_offsets'].active = False
    r_params['hla_angles'].active = False
    r_params['hla_offsets'].active = True
    r_params['hla_offsets'].target_val = torch.zeros(3)
    r_params['hla_offsets'].lambda_r = 0.5

    hla_atm = stage['hla_atm']
    sa_atm = stage['sa_atm']
    r_params['hla_angles'].set_call_fn(get_atm_angle_mean_closure(hla_atm))
    r_params['hla_offsets'].set_call_fn(get_atm_offset_mean_closure(hla_atm))
    r_params['sa_angles'].set_call_fn(get_atm_angle_mean_closure(sa_atm))
    r_params['sa_offsets'].set_call_fn(get_atm_offset_mean_closure(sa_atm))

    print('Init hla_theta_ap', stage['hla_atm'].init_theta_ap)
    [print(name, rp) for name, rp in r_params.items()]
    print()



def optimize_sa_offsets(stage):
    if 'epoch_sa_angles_mean' in stage:
        stage['sa_atm'].set_init_theta_ap(stage['epoch_sa_angles_mean'])

    r_params = stage['r_params']
    r_params['sa_angles'].active = True
    r_params['sa_angles'].target_val = stage['epoch_sa_angles_mean']
    r_params['sa_angles'].lambda_r = 0.2
    r_params['sa_offsets'].active = False
    r_params['sa_offsets'].target_val = torch.zeros(3)
    r_params['sa_offsets'].lambda_r = 0.1

    hla_atm = stage['hla_atm']
    sa_atm = stage['sa_atm']

    r_params['hla_angles'].set_call_fn(get_atm_angle_mean_closure(hla_atm))
    r_params['hla_offsets'].set_call_fn(get_atm_offset_mean_closure(hla_atm))
    r_params['sa_angles'].set_call_fn(get_atm_angle_mean_closure(sa_atm))
    r_params['sa_offsets'].set_call_fn(get_atm_offset_mean_closure(sa_atm))

    print('Init sa_theta_ap', stage['sa_atm'].init_theta_ap)
    [print(name, rp) for name, rp in r_params.items()]
    print()



def optimize_hla_offsets(stage):
    if 'epoch_hla_angles_mean' in stage:
        stage['hla_atm'].set_init_theta_ap(stage['epoch_hla_angles_mean'])

    r_params = stage['r_params']
    r_params['hla_angles'].active = True
    r_params['hla_angles'].target_val = stage['epoch_hla_angles_mean']
    r_params['hla_angles'].lambda_r = 0.2
    r_params['hla_offsets'].active = False
    r_params['hla_offsets'].target_val = torch.zeros(3)
    r_params['hla_offsets'].lambda_r = 0.1

    hla_atm = stage['hla_atm']
    sa_atm = stage['sa_atm']

    r_params['hla_angles'].set_call_fn(get_atm_angle_mean_closure(hla_atm))
    r_params['hla_offsets'].set_call_fn(get_atm_offset_mean_closure(hla_atm))
    r_params['sa_angles'].set_call_fn(get_atm_angle_mean_closure(sa_atm))
    r_params['sa_offsets'].set_call_fn(get_atm_offset_mean_closure(sa_atm))

    print('Init hla_theta_ap', stage['hla_atm'].init_theta_ap)
    [print(name, rp) for name, rp in r_params.items()]