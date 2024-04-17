import random
import functools
import numpy as np
import torch
from pathlib import Path
import copy
import contextlib
import einops as eo
import gc
from collections import defaultdict



def get_module(module, keychain):
    get_fn = lambda self, key: self[int(key)] if isinstance(self, torch.nn.Sequential) \
                                              else getattr(self, key)
    return functools.reduce(get_fn, keychain.split('.'), module)

def set_module(module, keychain, replacee):
    """Replaces any module inside a pytorch module for a given keychain with "replacee".
       Use module.named_modules() to retrieve valid keychains for layers.
       e.g.
       first_keychain = list(module.keys())[0]
       new_first_replacee = torch.nn.Conv1d(1,2,3)
       set_module(first_keychain, torch.nn.Conv1d(1,2,3))
    """
    get_fn = lambda self, key: self[int(key)] if isinstance(self, torch.nn.Sequential) \
        else getattr(self, key)

    key_list = keychain.split('.')
    root = functools.reduce(get_fn, key_list[:-1], module)
    leaf_id = key_list[-1]

    if isinstance(root, torch.nn.Sequential) and leaf_id.isnumeric():
        root[int(leaf_id)] = replacee
    else:
        setattr(root, leaf_id, replacee)



def ensure_dense(label):
    entered_sparse = label.is_sparse
    if entered_sparse:
        label = label.to_dense()

    return label, entered_sparse



def reduce_label_scores_epoch(label_scores_epoch):
    scores = copy.deepcopy(label_scores_epoch)

    nanmean_per_label = copy.deepcopy(scores)
    std_per_label = copy.deepcopy(scores)

    # Reduce over samples -> score per labels
    for m_name, m_dict in scores.items():
        for tag in m_dict:
            nanmean_per_label[m_name][tag] = np.nanmean(m_dict[tag])
            std_per_label[m_name][tag] = np.std(m_dict[tag])

    # Reduce over all -> score per metric
    nanmean_over_all = {}
    std_over_all = {}

    for m_name, m_dict in scores.items():
        all_metric_values = []
        for tag in m_dict:
            vals = m_dict[tag]
            if not isinstance(vals, list):
                vals = list(vals)
            all_metric_values = all_metric_values + vals

        nanmean_over_all[m_name] = np.nanmean(all_metric_values)
        std_over_all[m_name] = np.std(all_metric_values)

    return nanmean_per_label, std_per_label, nanmean_over_all, std_over_all



def get_batch_score_per_label(label_scores_epoch, metric_name, b_score, label_tags, exclude_bg=True) -> dict:
    """
        Converts metric tensors of shape (B,C) to a dictionary.
        Scores are appended to label_scores_epoch.
    """
    assert 'background' in label_tags, "Always provide 'background' tag name. Omit it with 'exclude_bg=True'"

    for tag_idx, tag in enumerate(label_tags):
        if exclude_bg and tag_idx == 0:
            continue
        for val in b_score[:,tag_idx]:
            if torch.isnan(val).item():
                val = float('nan')
            else:
                val = val.item()

            score_dict = label_scores_epoch.get(metric_name, {})
            values = score_dict.get(tag, [])
            values += [val]
            score_dict[tag] = values
            label_scores_epoch[metric_name] = score_dict
    return label_scores_epoch



def get_batch_dice_over_all(b_dice, exclude_bg=True) -> float:

    start_idx = 1 if exclude_bg else 0
    if torch.all(torch.isnan(b_dice[:,start_idx:])):
        return float('nan')
    return np.nanmean(b_dice[:,start_idx:]).item()



def reset_determinism():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    # torch.use_deterministic_algorithms(True)



def save_model(_path, epx=None, loss=None, **statefuls):
    _path = Path(_path).resolve()
    _path.mkdir(exist_ok=True, parents=True)

    for name, stful in statefuls.items():
        if stful != None:
            st_dict = stful.state_dict()
            st_dict['metadata'] = dict(epx=epx, loss=loss)
            torch.save(st_dict, _path.joinpath(name+'.pth'))




def get_rotation_matrix_3d_from_angles(deg_angles):
    """3D rotation matrix."""
    angles = torch.deg2rad(deg_angles)
    ax, ay, az = angles
    d = deg_angles.device
    Rx_0 = torch.tensor([1, 0, 0], device=d)
    Rx_1 = torch.stack([torch.tensor(0, device=d), torch.cos(ax), -torch.sin(ax)])
    Rx_2 = torch.stack([torch.tensor(0, device=d), torch.sin(ax), torch.cos(ax)])
    Rx = torch.stack([Rx_0, Rx_1, Rx_2])

    Ry_0 = torch.stack([torch.cos(ay), torch.tensor(0, device=d), torch.sin(ay)])
    Ry_1 = torch.tensor([0, 1, 0], device=d)
    Ry_2 = torch.stack([-torch.sin(ay), torch.tensor(0, device=d), torch.cos(ay)])
    Ry = torch.stack([Ry_0, Ry_1, Ry_2])

    Rz_0 = torch.stack([torch.cos(az), -torch.sin(az), torch.tensor(0, device=d)])
    Rz_1 = torch.stack([torch.sin(az), torch.cos(az), torch.tensor(0, device=d)])
    Rz_2 = torch.tensor([0, 0, 1], device=d)
    Rz = torch.stack([Rz_0, Rz_1, Rz_2])

    return Rz @ Ry @ Rx



def get_bincounts(label, num_classes):
    bn_counts = torch.bincount(label.reshape(-1).long(), minlength=num_classes)
    return bn_counts



def get_named_layers_leaves(module):
    """ Returns all leaf layers of a pytorch module and a keychain as identifier.
        e.g.
        ...
        ('features.0.5', nn.ReLU())
        ...
        ('classifier.0', nn.BatchNorm2D())
        ('classifier.1', nn.Linear())
    """

    return [(keychain, sub_mod) for keychain, sub_mod in list(module.named_modules()) if not next(sub_mod.children(), None)]



@contextlib.contextmanager
def temp_forward_hooks(modules, pre_fwd_hook_fn=None, post_fwd_hook_fn=None):
    handles = []
    if pre_fwd_hook_fn:
        handles.extend([mod.register_forward_pre_hook(pre_fwd_hook_fn) for mod in modules])
    if post_fwd_hook_fn:
        handles.extend([mod.register_forward_hook(post_fwd_hook_fn) for mod in modules])

    yield
    for hand in handles:
        hand.remove()



def debug_forward_pass(module, inpt, STEP_MODE=False):
    named_leaves = get_named_layers_leaves(module)
    leave_mod_dict = {mod:keychain for keychain, mod in named_leaves}

    def get_shape_str(interface_var):
        if isinstance(interface_var, tuple):
            shps = [str(elem.shape) if isinstance(elem, torch.Tensor) else type(elem) for elem in interface_var]
            return ', '.join(shps)
        elif isinstance(interface_var, torch.Tensor):
            return interface_var.shape
        return type(interface_var)

    def print_pre_info(module, inpt):
        inpt_shapes = get_shape_str(inpt)
        print(f"in:  {inpt_shapes}")
        print(f"key: {leave_mod_dict[module]}")
        print(f"mod: {module}")
        if STEP_MODE:
            input("To continue forward pass press [ENTER]")

    def print_post_info(module, inpt, output):
        output_shapes = get_shape_str(output)
        print(f"out: {output_shapes}\n")

    with temp_forward_hooks(leave_mod_dict.keys(), print_pre_info, print_post_info):
        return module(inpt)



def anomaly_hook(self, _input, output):
    if isinstance(_input, tuple):
        _input = list(_input)
    elif isinstance(_input, dict):
        _input = _input.values()
    elif isinstance(_input, list):
        pass

    if isinstance(output, tuple):
        output = list(output)
    elif isinstance(output, dict):
        output = output.values()
    elif isinstance(output, list):
        pass

    for inp_idx, inp_item in enumerate(_input):
        if isinstance(inp_item, torch.Tensor):
            nan_mask = torch.isnan(inp_item)
            inf_mask = torch.isinf(inp_item)
            raise RuntimeError(f"Found nan/inf in input")

    for out_idx, out_item in enumerate(output):
        if isinstance(out_item, torch.Tensor):
            nan_mask = torch.isnan(out_item)
            inf_mask = torch.isinf(out_item)
            raise RuntimeError(f"Found nan/inf in output")



def print_torch_memory_vars(last_mem_dict=None, show_leaves=True, show_cpu=True, show_empty=False):
    mem_dict = defaultdict(lambda: 0)
    torch.cuda.empty_cache()

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if not obj.is_leaf and str(obj.device) == 'cuda:0':
                    mem_dict[obj.size()] +=1
                    # type(obj), obj.size(), obj.is_leaf, obj.device
        except:
            pass

    if last_mem_dict is not None:
        print("Torch memory footprint diff")

        for size_key in set(list(last_mem_dict.keys()) + list(mem_dict.keys())):
            num = mem_dict[size_key]
            if size_key in last_mem_dict:
                print(size_key, num, f"({num - last_mem_dict[size_key]:+})")
            elif size_key in mem_dict:
                print(size_key, num, f"({num:+})")
    else:
        print("Torch memory footprint")
        for size_key, num in mem_dict.items():
            print(size_key, f"#{num}")
    print()
    return mem_dict



def determine_network_output_size(net, _input):
    with torch.no_grad():
        return net(_input).shape



def cut_slice(b_volume):
    b_volume = eo.rearrange(b_volume, 'B C D H W -> W B C D H')
    center_idx = b_volume.shape[0]//2
    b_volume = b_volume[center_idx:center_idx+1]
    return eo.rearrange(b_volume, ' W B C D H -> B C D H W')



def soft_cut_slice(b_volume, std=50.0):
    b_volume = eo.rearrange(b_volume, 'B C D H W -> W B C D H')
    W = b_volume.shape[0]
    center_idx = W//2

    n_dist = torch.distributions.normal.Normal(torch.tensor(center_idx), torch.tensor(std))

    probs = torch.arange(0, W)
    probs = n_dist.log_prob(probs).exp()
    probs = probs / probs.max()
    probs = probs.to(b_volume.device)

    b_volume = (b_volume * probs.view(W,1,1,1,1)).sum(0, keepdim=True)

    return eo.rearrange(b_volume, ' W B C D H -> B C D H W')



def get_binarized_from_onehot_label(onehot_label):
    onehot_bg = onehot_label[:,0:1]
    onehot_fg = onehot_label[:,1:].sum(dim=1, keepdim=True)
    return torch.cat([onehot_bg, onehot_fg], dim=1)