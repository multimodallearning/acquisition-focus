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



def get_atm_angle_mean_closure(atm):
    def closure(target_val):
        return ((get_theta_params(atm.last_theta)[0][:,0]-target_val)**2).mean()
    return closure



def get_atm_angle_std_closure(atm):
    def closure(target_val):
        return (get_theta_params(atm.last_theta)[0][:,0].std()-target_val).abs()
    return closure



def get_atm_offset_mean_closure(atm):
    def closure(target_val):
        return ((get_theta_params(atm.last_theta_t)[1][:,1]-target_val)**2).mean()
    return closure



def get_atm_offset_std_closure(atm):
    def closure(target_val):
        return (get_theta_params(atm.last_theta_t)[1][:,1].std()-target_val).abs()
    return closure



def optimize_sa_angles(stage):
    if 'last_sa_theta_rd' in stage:
        stage['w_atm'].set_init_theta_ap(torch.tensor([stage['last_sa_theta_rd'],.0,.0]))

    r_params = stage['regularization_parameters']
    r_params['sa_theta_r'].active = False
    r_params['sa_theta_t'].active = True
    r_params['sa_theta_t'].target_val = 0.0
    r_params['sa_theta_t'].lambda_r = 0.5

    h_atm = stage['h_atm']
    w_atm = stage['w_atm']
    r_params['hla_theta_r'].set_call_fn(get_atm_angle_mean_closure(h_atm))
    r_params['hla_theta_t'].set_call_fn(get_atm_offset_mean_closure(h_atm))
    r_params['sa_theta_r'].set_call_fn(get_atm_angle_mean_closure(w_atm))
    r_params['sa_theta_t'].set_call_fn(get_atm_offset_mean_closure(w_atm))


    print('Init w_theta_a', stage['w_atm'].init_theta_ap)
    [print(name, rp) for name, rp in stage['regularization_parameters'].items()]



def optimize_hla_angles(stage):

    stage['regularization_parameters']['sa_theta_r'].active = False
    stage['regularization_parameters']['sa_theta_t'].active = False

    stage['regularization_parameters']['hla_theta_r'].active = False
    stage['regularization_parameters']['hla_theta_t'].active = True
    stage['regularization_parameters']['hla_theta_t'].target_val = 0.0
    stage['regularization_parameters']['hla_theta_t'].lambda_r = 0.5

    h_atm = stage['h_atm']
    w_atm = stage['w_atm']
    r_params['hla_theta_r'].set_call_fn(get_atm_angle_mean_closure(h_atm))
    r_params['hla_theta_t'].set_call_fn(get_atm_offset_mean_closure(h_atm))
    r_params['sa_theta_r'].set_call_fn(get_atm_angle_mean_closure(w_atm))
    r_params['sa_theta_t'].set_call_fn(get_atm_offset_mean_closure(w_atm))

    print('Init h_theta_a', stage['h_atm'].init_theta_ap)
    [print(name, rp) for name, rp in stage['regularization_parameters'].items()]



def optimize_w_offsets(stage):
    if 'last_sa_theta_rd' in stage:
        stage['w_atm'].set_init_theta_ap(torch.tensor([stage['last_sa_theta_rd'],.0,.0]))

    stage['regularization_parameters']['sa_theta_r'].active = True
    stage['regularization_parameters']['sa_theta_r'].target_val = stage['last_sa_theta_rd']
    stage['regularization_parameters']['sa_theta_r'].lambda_r = 0.2
    stage['regularization_parameters']['sa_theta_t'].target_val = 0.0
    stage['regularization_parameters']['sa_theta_t'].lambda_r = 0.1

    h_atm = stage['h_atm']
    w_atm = stage['w_atm']
    r_params['hla_theta_r'].set_call_fn(get_atm_angle_mean_closure(h_atm))
    r_params['hla_theta_t'].set_call_fn(get_atm_offset_mean_closure(h_atm))
    r_params['sa_theta_r'].set_call_fn(get_atm_angle_mean_closure(w_atm))
    r_params['sa_theta_t'].set_call_fn(get_atm_offset_mean_closure(w_atm))

    print('Init w_theta_a', stage['w_atm'].init_theta_ap)
    [print(name, rp) for name, rp in stage['regularization_parameters'].items()]



def optimize_h_offsets(stage):
    if 'last_hla_theta_rd' in stage:
        stage['h_atm'].set_init_theta_ap(torch.tensor([stage['last_hla_theta_rd'],.0,.0]))

    stage['regularization_parameters']['hla_theta_r'].active = True
    stage['regularization_parameters']['hla_theta_r'].target_val = stage['last_hla_theta_rd']
    stage['regularization_parameters']['hla_theta_r'].lambda_r = 0.2
    stage['regularization_parameters']['hla_theta_t'].target_val = 0.0
    stage['regularization_parameters']['hla_theta_t'].lambda_r = 0.1

    h_atm = stage['h_atm']
    w_atm = stage['w_atm']
    r_params['hla_theta_r'].set_call_fn(get_atm_angle_mean_closure(h_atm))
    r_params['hla_theta_t'].set_call_fn(get_atm_offset_mean_closure(h_atm))
    r_params['sa_theta_r'].set_call_fn(get_atm_angle_mean_closure(w_atm))
    r_params['sa_theta_t'].set_call_fn(get_atm_offset_mean_closure(w_atm))

    print('Init h_theta_a', stage['h_atm'].init_theta_ap)
    [print(name, rp) for name, rp in stage['regularization_parameters'].items()]