
from contextlib import contextmanager
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import functools
import copy
import contextlib
from scipy.ndimage import distance_transform_edt as distance
import einops as eo

MOD_GET_FN = lambda self, key: self[int(key)] if isinstance(self, nn.Sequential) \
                                              else getattr(self, key)

@contextmanager
def torch_manual_seeded(seed):
    saved_state = torch.get_rng_state()
    yield
    torch.set_rng_state(saved_state)



def ensure_dense(label):
    entered_sparse = label.is_sparse
    if entered_sparse:
        label = label.to_dense()

    return label, entered_sparse



def restore_sparsity(label, was_sparse):
    if was_sparse and not label.is_sparse:
        return label.to_sparse()
    return label



def dilate_label_class(b_label, class_max_idx, class_dilate_idx, kernel_sz=2):
    dims = b_label.dim()
    assert (dims in [4,5])

    B,C, *_ = b_label.shape

    if kernel_sz < 2:
        return b_label

    b_dilated_label = b_label

    b_onehot = torch.nn.functional.one_hot(b_label.long(), class_max_idx+1)
    class_slice = b_onehot[...,class_dilate_idx]

    if dims == 4:
        *_, H, W = class_slice.shape
        kernel = torch.ones([kernel_sz,kernel_sz]).long()
        kernel = kernel.view(1,1,kernel_sz,kernel_sz)
        class_slice = torch.nn.functional.conv2d(
            class_slice.view(B,1,H,W), kernel, padding='same')

    elif dims == 5:
        *_, D, H, W = class_slice.shape
        kernel = torch.ones([kernel_sz,kernel_sz,kernel_sz])
        kernel = kernel.long().view(1,1,kernel_sz,kernel_sz,kernel_sz)
        class_slice = torch.nn.functional.conv3d(
            class_slice.view(B,1,D,H,W), kernel, padding='same')

    else:
        raise ValueError()

    dilated_class_slice = torch.clamp(class_slice, 0, 1)
    b_dilated_label[dilated_class_slice.bool()] = class_dilate_idx

    return b_dilated_label



def interpolate_sample(b_image=None, b_label=None, scale_factor=1.,
                       use_2d=False):
    if use_2d:
        scale = [scale_factor]*2
        im_mode = 'bilinear'
    else:
        scale = [scale_factor]*3
        im_mode = 'trilinear'

    if b_image is not None and b_image.numel() > 0:
        b_image = F.interpolate(
            b_image.unsqueeze(1), scale_factor=scale, mode=im_mode, align_corners=True,
            recompute_scale_factor=False
        )
        b_image = b_image.squeeze(1)

    if b_label is not None and b_label.numel() > 0:
        b_label = F.interpolate(
            b_label.unsqueeze(1).float(), scale_factor=scale, mode='nearest',
            recompute_scale_factor=False
        ).long()
        b_label = b_label.squeeze(1)

    return b_image, b_label



def augment_noise(b_image, strength=0.05):
    return b_image + strength*torch.randn_like(b_image)



def spatial_augment(b_image=None, b_label=None,
    bspline_num_ctl_points=6, bspline_strength=0.005, bspline_probability=.9,
    affine_strength=0.08, add_affine_translation=0., affine_probability=.45,
    pre_interpolation_factor=None,
    use_2d=False,
    b_grid_override=None):

    """
    2D/3D b-spline augmentation on image and segmentation mini-batch on GPU.
    :input: b_image batch (torch.cuda.FloatTensor), b_label batch (torch.cuda.LongTensor)
    :return: augmented Bx(D)xHxW image batch (torch.cuda.FloatTensor),
    augmented Bx(D)xHxW seg batch (torch.cuda.LongTensor)
    """

    do_bspline = (np.random.rand() < bspline_probability)
    do_affine = (np.random.rand() < affine_probability)

    if pre_interpolation_factor:
        b_image, b_label = interpolate_sample(b_image, b_label, pre_interpolation_factor, use_2d)

    KERNEL_SIZE = 3

    if b_image is None and b_label.numel() > 0:
        common_shape = b_label.shape
        common_device = b_label.device

    elif b_label is None and b_image.numel() > 0:
        common_shape = b_image.shape
        common_device = b_image.device

    else:
        assert b_image.shape == b_label.shape, \
            f"Image and label shapes must match but are {b_image.shape} and {b_label.shape}."
        common_shape = b_image.shape
        common_device = b_image.device

    if b_grid_override is None:
        if use_2d:
            assert len(common_shape) == 3, \
                f"Augmenting 2D. Input batch " \
                f"should be BxHxW but is {common_shape}."
            B,H,W = common_shape

            identity = torch.eye(2,3).expand(B,2,3).to(common_device)
            id_grid = F.affine_grid(identity, torch.Size((B,2,H,W)),
                align_corners=False)

            grid = id_grid

            if do_bspline:
                bspline = torch.nn.Sequential(
                    torch.nn.AvgPool2d(KERNEL_SIZE,stride=1,padding=int(KERNEL_SIZE//2)),
                    torch.nn.AvgPool2d(KERNEL_SIZE,stride=1,padding=int(KERNEL_SIZE//2)),
                    torch.nn.AvgPool2d(KERNEL_SIZE,stride=1,padding=int(KERNEL_SIZE//2))
                ).to(common_device)
                # Add an extra *.5 factor to dim strength to make strength fit 3D case
                dim_strength = (torch.tensor([H,W]).float()*bspline_strength*.5).to(common_device)
                rand_control_points = dim_strength.view(1,2,1,1) * \
                    (
                        torch.randn(B, 2, bspline_num_ctl_points, bspline_num_ctl_points)
                    ).to(common_device)

                bspline_disp = bspline(rand_control_points)
                bspline_disp = torch.nn.functional.interpolate(
                    bspline_disp, size=(H,W), mode='bilinear', align_corners=True
                ).permute(0,2,3,1)

                grid += bspline_disp

            if do_affine:
                affine_matrix = (torch.eye(2,3).unsqueeze(0) + \
                    affine_strength * torch.randn(B,2,3)).to(common_device)
                # Add additional x,y offset
                alpha = np.random.rand() * 2 * np.pi
                offset_dir =  torch.tensor([np.cos(alpha), np.sin(alpha)])
                affine_matrix[:,:,-1] = add_affine_translation * offset_dir
                affine_disp = F.affine_grid(affine_matrix, torch.Size((B,1,H,W)),
                                        align_corners=False)
                grid += (affine_disp-id_grid)

        else:
            assert len(common_shape) == 4, \
                f"Augmenting 3D. Input batch " \
                f"should be BxDxHxW but is {common_shape}."
            B,D,H,W = common_shape

            identity = torch.eye(3,4).expand(B,3,4).to(common_device)
            id_grid = F.affine_grid(identity, torch.Size((B,3,D,H,W)),
                align_corners=False)

            grid = id_grid

            if do_bspline:
                bspline = torch.nn.Sequential(
                    torch.nn.AvgPool3d(KERNEL_SIZE,stride=1,padding=int(KERNEL_SIZE//2)),
                    torch.nn.AvgPool3d(KERNEL_SIZE,stride=1,padding=int(KERNEL_SIZE//2)),
                    torch.nn.AvgPool3d(KERNEL_SIZE,stride=1,padding=int(KERNEL_SIZE//2))
                ).to(b_image.device)
                dim_strength = (torch.tensor([D,H,W]).float()*bspline_strength).to(common_device)

                rand_control_points = dim_strength.view(1,3,1,1,1)  * \
                    (
                        torch.randn(B, 3, bspline_num_ctl_points, bspline_num_ctl_points, bspline_num_ctl_points)
                    ).to(b_image.device)

                bspline_disp = bspline(rand_control_points)

                bspline_disp = torch.nn.functional.interpolate(
                    bspline_disp, size=(D,H,W), mode='trilinear', align_corners=True
                ).permute(0,2,3,4,1)

                grid += bspline_disp

            if do_affine:
                affine_matrix = (torch.eye(3,4).unsqueeze(0) + \
                    affine_strength * torch.randn(B,3,4)).to(common_device)

                # Add additional x,y,z offset
                theta = np.random.rand() * 2 * np.pi
                phi = np.random.rand() * 2 * np.pi
                offset_dir =  torch.tensor([
                    np.cos(phi)*np.sin(theta),
                    np.sin(phi)*np.sin(theta),
                    np.cos(theta)])
                affine_matrix[:,:,-1] = add_affine_translation * offset_dir

                affine_disp = F.affine_grid(affine_matrix,torch.Size((B,1,D,H,W)),
                                        align_corners=False)

                grid += (affine_disp-id_grid)
    else:
        # Override grid with external value
        grid = b_grid_override

    if b_image is not None:
        b_image_out = F.grid_sample(
            b_image.unsqueeze(1).float(), grid,
            padding_mode='border', align_corners=False)
        b_image_out = b_image_out.squeeze(1)
    else:
        b_image_out = None

    if b_label is not None:
        b_label_out = F.grid_sample(
            b_label.unsqueeze(1).float(), grid,
            mode='nearest', align_corners=False)
        b_label_out = b_label_out.squeeze(1).long()
    else:
        b_label_out = None

    b_out_grid = grid


    return b_image_out, b_label_out, b_out_grid



def get_seg_metrics_per_label(label_scores_epoch, b_input, b_target, label_tags, spacing,
    selected_metrics=('dice', 'jaccard', 'precision', 'recall', 'fpr', 'fnr', 'vs', 'hd', 'hd95', 'msd', 'mdsd','stdsd'),
    exclude_bg=True):

    assert type(spacing) == np.ndarray and len(spacing) == 3
    assert b_input.dim() == 5 and b_target.dim() == 5
    assert len(label_tags) == b_input.shape[-1] == b_target.shape[-1]
    assert 'background' in label_tags, "Always provide 'background' tag name. Omit it with 'exclude_bg=True'"

    b_input = b_input.cpu().numpy()
    b_target = b_target.cpu().numpy()
    B, *_ = b_input.shape

    b_metrics = []
    start_lbl_idx = 1 if exclude_bg else 1

    for b_idx in range(B):
        raise NotImplementedError()
        md = sg.get_metrics_dict_all_labels(range(start_lbl_idx, len(label_tags)), b_input[b_idx], b_target[b_idx],
            spacing=spacing, metrics_names=selected_metrics
        )
        del md['label']
        b_metrics.append(md)


    label_tags = list(label_tags)

    if exclude_bg:
        label_tags = label_tags[1:]

    # Resort metrics: Metrics are inserted by label tag for each metric name
    for tag in label_tags:
        tag_idx = label_tags.index(tag)
        for metrics in b_metrics:
            for m_name, m_values in metrics.items():
                label_metrics = label_scores_epoch.get(m_name, {})
                tag_metric = label_metrics.get(tag, []) + [m_values[tag_idx]]
                label_metrics[tag] = tag_metric
                label_scores_epoch[m_name] = label_metrics
    return label_scores_epoch

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



def get_2d_stack_batch_size(b_input_size: torch.Size, stack_dim):
    assert len(b_input_size) == 5, f"Input size must be 5D: BxCxDxHxW but is {b_input_size}"
    if stack_dim == "D":
        return b_input_size[0]*b_input_size[2]
    if stack_dim == "H":
        return b_input_size[0]*b_input_size[3]
    if stack_dim == "W":
        return b_input_size[0]*b_input_size[4]
    else:
        raise ValueError(f"stack_dim '{stack_dim}' must be 'D' or 'H' or 'W'.")



def make_2d_stack_from_3d(b_input, stack_dim):
    assert b_input.dim() == 5, f"Input must be 5D: BxCxDxHxW but is {b_input.shape}"
    B, C, D, H, W = b_input.shape

    if stack_dim == "D":
        return b_input.permute(0, 2, 1, 3, 4).reshape(B*D, C, H, W)
    if stack_dim == "H":
        return b_input.permute(0, 3, 1, 2, 4).reshape(B*H, C, D, W)
    if stack_dim == "W":
        return b_input.permute(0, 4, 1, 2, 3).reshape(B*W, C, D, H)
    else:
        raise ValueError(f"stack_dim '{stack_dim}' must be 'D' or 'H' or 'W'.")



def make_3d_from_2d_stack(b_input, stack_dim, orig_stack_size):
    assert b_input.dim() == 4, f"Input must be 4D: (orig_batch_size/B)xCxSPAT1xSPAT0 but is {b_input.shape}"
    B, C, SPAT1, SPAT0 = b_input.shape
    b_input = b_input.reshape(orig_stack_size, int(B//orig_stack_size), C, SPAT1, SPAT0)

    if stack_dim == "D":
        return b_input.permute(0, 2, 1, 3, 4)
    if stack_dim == "H":
        return b_input.permute(0, 2, 3, 1, 4)
    if stack_dim == "W":
        return b_input.permute(0, 2, 3, 4, 1)
    else:
        raise ValueError(f"stack_dim is '{stack_dim}' but must be 'D' or 'H' or 'W'.")



def get_module(module, keychain):
    """Retrieves any module inside a pytorch module for a given keychain.
       module.named_ to retrieve valid keychains for layers.
    """

    return functools.reduce(MOD_GET_FN, keychain.split('.'), module)



def set_module(module, keychain, replacee):
    """Replaces any module inside a pytorch module for a given keychain with "replacee".
       Use module.named_modules() to retrieve valid keychains for layers.
       e.g.
       first_keychain = list(module.keys())[0]
       new_first_replacee = torch.nn.Conv1d(1,2,3)
       set_module(first_keychain, torch.nn.Conv1d(1,2,3))
    """
    MOD_GET_FN = lambda self, key: self[int(key)] if isinstance(self, torch.nn.Sequential) \
        else getattr(self, key)

    key_list = keychain.split('.')
    root = functools.reduce(MOD_GET_FN, key_list[:-1], module)
    leaf_id = key_list[-1]

    if isinstance(root, torch.nn.Sequential) and leaf_id.isnumeric():
        root[int(leaf_id)] = replacee
    else:
        setattr(root, leaf_id, replacee)




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



def get_test_func_all_parameters_updated():
    was_checked = False

    def test_all_parameters_updated(model):
        if was_checked: return
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.equal(torch.sum(param.grad ** 2), torch.tensor(0.).to(param.device))

    return test_all_parameters_updated



def calc_dist_map(onehot_seg, mode='signed'):
    assert mode in ['inner', 'outer', 'signed', 'unsigned']
    assert onehot_seg.dim() == 5
    B,C,D,H,W = onehot_seg.shape
    assert onehot_seg.dtype == torch.bool
    onehot_seg = eo.rearrange(onehot_seg, 'B C D H W -> (B C) D H W')
    # see https://github.com/LIVIAETS/boundary-loss
    dm = torch.zeros_like(onehot_seg, dtype=torch.float)
    for idx, chan in enumerate(onehot_seg):
        posmask = chan.numpy()
        negmask = ~posmask
        if posmask.any():
            if mode == 'inner':
                dm[idx] = -torch.as_tensor(distance(posmask))
            elif mode == 'outer':
                dm[idx] = torch.as_tensor(distance(negmask))
            elif mode == 'signed':
                dm[idx] = torch.as_tensor(distance(negmask) * negmask - (distance(posmask) - 1) * posmask)
            elif mode == 'unsigned':
                dm[idx] = torch.as_tensor(distance(negmask) * negmask + (distance(posmask) - 1) * posmask)

    dm = dm.view(B,C,D,H,W)
    return dm


def get_seg_boundary(binary_seg):
    assert binary_seg.dtype == torch.bool
    fg_dil = dilate_label_class(binary_seg.float(),1,1, kernel_sz=3)
    bg_dil = dilate_label_class(binary_seg.float(),1,0, kernel_sz=3)

    fg_cont = torch.logical_xor(fg_dil.bool(), binary_seg)
    bg_cont = torch.logical_xor(bg_dil.bool(), binary_seg)

    return torch.logical_or(fg_cont, bg_cont)



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
            nan_mask = torch.isnan(out)
            inf_mask = torch.isinf(out)
            raise RuntimeError(f"Found nan/inf in output")