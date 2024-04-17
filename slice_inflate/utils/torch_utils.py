import random
import functools
import numpy as np
import torch
from pathlib import Path
import copy



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



def get_binarized_from_onehot_label(onehot_label):
    onehot_bg = onehot_label[:,0:1]
    onehot_fg = onehot_label[:,1:].sum(dim=1, keepdim=True)
    return torch.cat([onehot_bg, onehot_fg], dim=1)



class NoneOptimizer():
    def __init__(self):
        super().__init__()
    def step(self):
        pass
    def zero_grad(self):
        pass
    def state_dict(self):
        return {}