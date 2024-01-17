# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: 'Python 3.9.13 (''.venv'': poetry)'
#     language: python
#     name: python3
# ---

# %%
import os
import sys
import re
from pathlib import Path
import json
import dill
import einops as eo
from datetime import datetime
from git import Repo
import joblib
import copy
from enum import Enum

from slice_inflate.utils.common_utils import get_script_dir
THIS_SCRIPT_DIR = get_script_dir()

os.environ['CACHE_PATH'] = str(Path(THIS_SCRIPT_DIR, '.cache'))

from meidic_vtach_utils.run_on_recommended_cuda import get_cuda_environ_vars as get_vars
os.environ.update(get_vars('*'))

import torch
torch.set_printoptions(sci_mode=False)
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb
import nibabel as nib

import contextlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from slice_inflate.utils.nifti_utils import crop_around_label_center
from slice_inflate.utils.log_utils import get_global_idx, log_label_metrics, \
    log_oa_metrics, log_affine_param_stats, log_frameless_image, get_cuda_mem_info_str
from slice_inflate.datasets.clinical_cardiac_views import get_class_volumes
from sklearn.model_selection import KFold
import numpy as np
import monai

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from slice_inflate.datasets.mmwhs_dataset import MMWHSDataset
from slice_inflate.datasets.mrxcat_dataset import MRXCATDataset

from slice_inflate.utils.common_utils import DotDict, in_notebook
from slice_inflate.utils.torch_utils import reset_determinism, ensure_dense, \
    get_batch_dice_over_all, get_batch_score_per_label, save_model, \
    reduce_label_scores_epoch, get_test_func_all_parameters_updated, anomaly_hook, cut_slice
from slice_inflate.models.nnunet_models import Generic_UNet_Hybrid
from slice_inflate.models.learnable_transform import AffineTransformModule, SoftCutModule, HardCutModule, get_random_ortho6_vector
from slice_inflate.models.ae_models import BlendowskiAE, BlendowskiVAE, HybridAE
from slice_inflate.losses.regularization import optimize_sa_angles, optimize_sa_offsets, optimize_hla_angles, optimize_hla_offsets, init_regularization_params, deactivate_r_params, Stage, StageIterator
from slice_inflate.utils.nnunetv2_utils import get_segment_fn
from slice_inflate.utils.nifti_utils import get_zooms
from slice_inflate.models.nnunet_models import SkipConnector
from slice_inflate.utils.nifti_utils import nifti_grid_sample

NOW_STR = datetime.now().strftime("%Y%m%d__%H_%M_%S")
THIS_REPO = Repo(THIS_SCRIPT_DIR)
PROJECT_NAME = "slice_inflate"

training_dataset, test_dataset = None, None
test_all_parameters_updated = get_test_func_all_parameters_updated()
# %%

with open(Path(THIS_SCRIPT_DIR, 'config_dict.json'), 'r') as f:
    config_dict = DotDict(json.load(f))

# Log commmit id and dirtiness
dirty_str = "!dirty-" if THIS_REPO.is_dirty() else ""
config_dict['git_commit'] = f"{dirty_str}{THIS_REPO.commit().hexsha}"

def prepare_data(config):
    args = [config.dataset[1]]

    if config.dataset[0] == 'mmwhs':
        dataset_class = MMWHSDataset
    elif config.dataset[0] == 'mrxcat':
        dataset_class = MRXCATDataset
    else:
        raise ValueError()

    kwargs = {k:v for k,v in config.items()}

    arghash = joblib.hash(joblib.hash(args)+joblib.hash(kwargs))
    cache_path = Path(os.environ['CACHE_PATH'], arghash, 'dataset.dil')
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.is_file():
        print("Loading dataset from cache:", cache_path)
        with open(cache_path, 'rb') as file:
            dataset = dill.load(file)
    else:
        dataset = dataset_class(*args, **kwargs)
        with open(cache_path, 'wb') as file:
            dill.dump(dataset, file)

    return dataset


# %%

run_test_once_only = not (config_dict.test_only_and_output_to in ["", None])

if training_dataset is None:
    train_config = DotDict(config_dict.copy())
    if run_test_once_only:
        train_config['state'] = 'empty'
    training_dataset = prepare_data(train_config)

if test_dataset is None:
    test_config = DotDict(config_dict.copy())
    test_config['state'] = 'test'
    test_dataset = prepare_data(test_config)

# %%


def get_norms(model):
    norms = {}
    for name, mod in model.named_modules():
        mod_norm = 0
        for p in mod.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                mod_norm += param_norm.item() ** 2
        mod_norm = mod_norm ** 0.5
        norms[name] = mod_norm
    return norms



def get_model(config, dataset_len, num_classes, THIS_SCRIPT_DIR, _path=None, load_model_only=False, encoder_training_only=False):

    device = config.device
    assert config.model_type in ['vae', 'ae', 'hybrid-ae', 'unet', 'hybrid-unet', 'unet-wo-skip', 'hybrid-unet-wo-skip']
    if not _path is None:
        _path = Path(THIS_SCRIPT_DIR).joinpath(_path).resolve()

    if config.model_type == 'vae':
        model = BlendowskiVAE(std_max=10.0, epoch=0, epoch_reach_std_max=250,
            in_channels=num_classes, out_channels=num_classes)

    elif config.model_type == 'ae':
        model = BlendowskiAE(in_channels=num_classes, out_channels=num_classes)

    elif config.model_type == 'hybrid-ae':
        model = HybridAE(in_channels=num_classes*2, out_channels=num_classes)

    elif 'unet' in config.model_type:
        assert config.model_type
        init_dict_path = Path(THIS_SCRIPT_DIR, "./slice_inflate/models/nnunet_init_dict_128_128_128.pkl")
        with open(init_dict_path, 'rb') as f:
            init_dict = dill.load(f)
        init_dict['num_classes'] = num_classes
        init_dict['deep_supervision'] = False
        init_dict['final_nonlin'] = torch.nn.Identity()

        use_skip_connections = True if not 'wo-skip' in config.model_type else False
        if 'hybrid' in config.model_type:
            enc_mode = '2d'
            dec_mode = '3d'
            init_dict['use_onehot_input'] = False
            init_dict['input_channels'] = num_classes*2
            init_dict['pool_op_kernel_sizes'][-1] = [2,2,2]
            init_dict['norm_op'] = nn.InstanceNorm3d
            nnunet_model = Generic_UNet_Hybrid(**init_dict, use_skip_connections=use_skip_connections, encoder_mode=enc_mode, decoder_mode=dec_mode)
        else:
            enc_mode = '3d'
            dec_mode = '3d'
            init_dict['use_onehot_input'] = True
            # nnunet_model = Generic_UNet(**init_dict, use_skip_connections=use_skip_connections)
            nnunet_model = Generic_UNet_Hybrid(**init_dict, use_skip_connections=use_skip_connections, encoder_mode=enc_mode, decoder_mode=dec_mode)

        seg_outputs = list(filter(lambda elem: 'seg_outputs' in elem[0], nnunet_model.named_parameters()))
        # Disable gradients of non-used deep supervision
        for so_idx in range(len(seg_outputs)-1):
            seg_outputs[so_idx][1].requires_grad = False

        class InterfaceModel(torch.nn.Module):
            def __init__(self, nnunet_model):
                super().__init__()
                self.nnunet_model = nnunet_model

            def forward(self, *args, **kwargs):
                y_hat = self.nnunet_model(*args, **kwargs)
                if isinstance(y_hat, tuple):
                    return y_hat[0]
                else:
                    return y_hat

        model = InterfaceModel(nnunet_model)

    else:
        raise ValueError

    model.to(device)

    if _path:
        assert Path(_path).is_dir()
        model_dict = torch.load(Path(_path).joinpath('model.pth'), map_location=device)
        epx = model_dict.get('metadata', {}).get('epx', 0)
        print(f"Loading model from {_path}")
        print(model.load_state_dict(model_dict, strict=False))

    else:
        print(f"Generating fresh '{type(model).__name__}' model.")
        epx = 0

    if encoder_training_only:
        decoder_modules = filter(lambda elem: 'decoder' in elem[0], model.named_modules())
        for nmod in decoder_modules:
            for param in nmod[1].parameters():
                param.requires_grad = False

    print(f"Trainable param count model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Non-trainable param count model: {sum(p.numel() for p in model.parameters() if not p.requires_grad)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scaler = amp.GradScaler()

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)



    if _path and not load_model_only:
        assert Path(_path).is_dir()
        print(f"Loading optimizer, scheduler, scaler from {_path}")
        optimizer.load_state_dict(torch.load(Path(_path).joinpath('optimizer.pth'), map_location=device))
        scheduler.load_state_dict(torch.load(Path(_path).joinpath('scheduler.pth'), map_location=device))
        scaler.load_state_dict(torch.load(Path(_path).joinpath('scaler.pth'), map_location=device))

    else:
        print(f"Generated fresh optimizer, scheduler, scaler.")

    # for submodule in model.modules():
    #     submodule.register_forward_hook(anomaly_hook)
    # for submodule in sa_atm.modules():
    #     submodule.register_forward_hook(anomaly_hook)
    # for submodule in hla_atm.modules():
    #     submodule.register_forward_hook(anomaly_hook)

    return (model, optimizer, scheduler, scaler), epx



def get_atm(config, num_classes, size_3d, view, this_script_dir, _path=None, random_ap_init=False):

    assert view in ['sa', 'hla']
    device = config.device


    # Add atm models
    atm = AffineTransformModule(num_classes,
        size_3d,
        torch.tensor(config.hires_fov_mm),
        torch.tensor(config.hires_fov_vox),
        offset_clip_value=config['offset_clip_value'],
        zoom_clip_value=config['zoom_clip_value'],
        optim_method=config.affine_theta_optim_method,
        tag=view)

    if random_ap_init:
        atm.set_init_theta_ap(get_random_ortho6_vector(rotation_strength=0.5, constrained=False))

    if _path:
        assert Path(_path).is_dir()
        atm_dict = torch.load(Path(_path).joinpath(f'{view}_atm.pth'), map_location=device)
        print(f"Loading {view} atm from {_path}")
        print(atm.load_state_dict(atm_dict, strict=False))

    return atm

class NoneOptimizer():
    def __init__(self):
        super().__init__()
    def step(self):
        pass
    def zero_grad(self):
        pass
    def state_dict(self):
        return {}

def get_transform_model(config, num_classes, size_3d, this_script_dir, _path=None, sa_atm_override=None, hla_atm_override=None):
    device = config.device

    if isinstance(sa_atm_override, AffineTransformModule):
        # Check if atm is set externally
        sa_atm = sa_atm_override
    else:
        sa_atm = get_atm(config, num_classes, size_3d, random_ap_init=config.use_random_affine_ap_init_sa, view='sa', this_script_dir=this_script_dir, _path=_path)

    if isinstance(hla_atm_override, AffineTransformModule):
        # Check if atm is set externally
        hla_atm = hla_atm_override
    else:
        hla_atm = get_atm(config, num_classes, size_3d, random_ap_init=config.use_random_affine_ap_init_hla, view='hla', this_script_dir=this_script_dir, _path = _path)

    if config['soft_cut_std'] > 0:
        sa_cut_module = SoftCutModule(soft_cut_softness=config['soft_cut_std'])
        hla_cut_module = SoftCutModule(soft_cut_softness=config['soft_cut_std'])
    else:
        sa_cut_module = HardCutModule()
        hla_cut_module = HardCutModule()

    sa_atm.to(device)
    hla_atm.to(device)
    sa_cut_module.to(device)
    hla_cut_module.to(device)

    if config.cuts_mode == 'sa':
        transform_parameters = list(sa_atm.parameters()) + list(sa_cut_module.parameters())
    elif config.cuts_mode == 'hla':
        transform_parameters = list(hla_atm.parameters()) + list(hla_cut_module.parameters())
    elif config.cuts_mode == 'sa>hla':
        transform_parameters = list(hla_atm.parameters()) + list(hla_cut_module.parameters())
    elif config.cuts_mode == 'sa+hla':
       transform_parameters = (
            list(sa_atm.parameters())
            + list(hla_atm.parameters())
            + list(sa_cut_module.parameters())
            + list(hla_cut_module.parameters())
        )
    else:
        raise ValueError()

    if config.train_affine_theta:
        assert config.use_affine_theta

    if config.train_affine_theta:
        transform_optimizer = torch.optim.AdamW(transform_parameters, weight_decay=0.1, lr=config.lr*2)
        transform_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(transform_optimizer, T_0=int(config.epochs/4))
    else:
        transform_optimizer = NoneOptimizer()
        transform_scheduler = None

    if _path and config.train_affine_theta:
        assert Path(_path).is_dir()
        print(f"Loading transform optimizer from {_path}")
        transform_optimizer.load_state_dict(torch.load(Path(_path).joinpath('transform_optimizer.pth'), map_location=device))

    else:
        print(f"Generated fresh transform optimizer.")

    return (sa_atm, hla_atm, sa_cut_module, hla_cut_module), transform_optimizer, transform_scheduler



# %%
def get_transformed(config, label, soft_label, nifti_affine, grid_affine_pre_mlp, hidden_augment_affine,
                    atm, cut_module, image=None, segment_fn=None):

    img_is_invalid = image is None or image.dim() == 0

    B, num_classes, D, H, W = label.shape

    if img_is_invalid:
        image = torch.zeros(B,1,D,H,W, device=label.device)

    # Transform  label with 'bilinear' interpolation to have gradients
    soft_label, label, grid_affine, _ = atm(soft_label.view(B, num_classes, D, H, W), label.view(B, num_classes, D, H, W),
        nifti_affine, grid_affine_pre_mlp, hidden_augment_affine)

    image, _, _, _ = atm(image.view(B, 1, D, H, W), None,
        nifti_affine, grid_affine_pre_mlp, hidden_augment_affine, theta_override=atm.last_theta)

    label_slc = cut_module(soft_label)
    image_slc = cut_module(image)

    if config.label_slice_type == 'from-gt':
        pass
    elif config.label_slice_type == 'from-segmented-hires':
        assert not img_is_invalid and segment_fn is not None
        # Beware: Label slice does not have gradients anymore
        pred_slc = segment_fn(eo.rearrange(image_slc, 'B C D H 1 -> B C 1 D H'), get_zooms(nifti_affine)).view(B,1,D,H)
        pred_slc = eo.rearrange(pred_slc, 'B 1 D H -> B D H 1').long()
        label_slc = eo.rearrange(F.one_hot(pred_slc, num_classes),
            'B D H 1 OH -> B OH D H 1')
        # plt.imshow(image_slc[0].squeeze().cpu(), cmap='gray')
        # plt.imshow(label_slc[0].argmax(1).squeeze().cpu(), cmap='magma', alpha=.5, interpolation='none')
        # plt.savefig('slice_seg.png')

    if img_is_invalid:
        image = torch.empty([])
        image_slc = torch.empty([])

    # Do not set label_slc to .long() here, since we (may) need the gradients
    return image, label.long(), image_slc, label_slc, grid_affine



def get_model_input(batch, config, num_classes, sa_atm, hla_atm, sa_cut_module, hla_cut_module, segment_fn):
    b_label = batch['label']
    b_image = batch['image']

    if config.clinical_view_affine_type == 'from-gt':
        b_view_affines = batch['additional_data']['gt_view_affines']
    elif config.clinical_view_affine_type == 'from-segmented-lores-prescan':
        b_view_affines = batch['additional_data']['lores_prescan_view_affines']

    b_label = eo.rearrange(F.one_hot(b_label, num_classes),
                        'B D H W OH -> B OH D H W')
    B,NUM_CLASSES,D,H,W = b_label.shape

    if config.use_distance_map_localization:
        b_soft_label = batch['additional_data']['label_distance_map']
    else:
        b_soft_label = b_label

    nifti_affine = batch['additional_data']['nifti_affine'].float()
    known_augment_affine = batch['additional_data']['known_augment_affine']
    hidden_augment_affine = batch['additional_data']['hidden_augment_affine']

    sa_atm.use_affine_theta = config.use_affine_theta
    hla_atm.use_affine_theta = config.use_affine_theta

    if 'sa' in config.cuts_mode:
        ctx = torch.no_grad \
            if config.cuts_mode == 'sa>hla' else contextlib.nullcontext # Do not use gradients when just inferring from SA view
        with ctx():
            # Use case dependent grid affine of p2CH view
            sa_input_grid_affine = torch.as_tensor(b_view_affines['p2CH']).view(B,4,4).to(known_augment_affine)

            sa_image, sa_label, sa_image_slc, sa_label_slc, sa_grid_affine = \
                get_transformed(
                    config,
                    b_label.view(B, NUM_CLASSES, D, H, W),
                    b_soft_label.view(B, NUM_CLASSES, D, H, W),
                    nifti_affine,
                    known_augment_affine, # @ sa_input_grid_affine,
                    hidden_augment_affine,
                    sa_atm, sa_cut_module,
                    image=b_image.view(B, 1, D, H, W), segment_fn=segment_fn)

    if 'hla' in config.cuts_mode:
        # Use case dependent grid affine of p2CH view
        hla_input_grid_affine = torch.as_tensor(b_view_affines['p4CH']).view(B,4,4).to(known_augment_affine)
        hla_image, hla_label, hla_image_slc, hla_label_slc, hla_grid_affine = \
            get_transformed(
                config,
                b_label.view(B, NUM_CLASSES, D, H, W),
                b_soft_label.view(B, NUM_CLASSES, D, H, W),
                nifti_affine,
                known_augment_affine, # @ hla_input_grid_affine,
                hidden_augment_affine,
                hla_atm, hla_cut_module,
                image=b_image.view(B, 1, D, H, W), segment_fn=segment_fn)

    if config.cuts_mode == 'sa':
        slices = [sa_label_slc, sa_label_slc]
        grid_affines = [sa_grid_affine, sa_grid_affine]
    elif config.cuts_mode == 'hla':
        slices = [hla_label_slc, hla_label_slc]
        grid_affines = [hla_grid_affine, hla_grid_affine]
    elif config.cuts_mode == 'sa>hla':
        slices = [sa_label_slc.detach(), hla_label_slc]
        grid_affines = [sa_grid_affine, hla_grid_affine]
    elif config.cuts_mode == 'sa+hla':
        slices = [sa_label_slc, hla_label_slc]
        grid_affines = [sa_grid_affine, hla_grid_affine]
    else:
        raise ValueError()

    SPAT = sa_label.shape[-1]

    if 'hybrid' in config.model_type:
        b_input = torch.cat(slices, dim=1)
        b_input = b_input.view(-1, NUM_CLASSES*2, SPAT, SPAT)
    else:
        b_input = torch.cat(slices, dim=-1)
        b_input = torch.cat([b_input] * int(SPAT/b_input.shape[-1]), dim=-1) # Stack data hla/sa next to each other

    if config.reconstruction_target == 'from-dataloader':
        b_target = b_label
    elif config.reconstruction_target == 'sa-oriented':
        raise ValueError("Currently not working together zoom parameters and LearnCutModule")
        b_target = sa_label
        grid_affines[0] = sa_grid_affine.inverse() @ grid_affines[0]
        grid_affines[1] = sa_grid_affine.inverse() @ grid_affines[1]
    elif config.reconstruction_target == 'hla-oriented':
        raise ValueError("Currently not working together zoom parameters and LearnCutModule")
        b_target = hla_label
        grid_affines[0] = hla_grid_affine.inverse() @ grid_affines[0]
        grid_affines[1] = hla_grid_affine.inverse() @ grid_affines[1]
    else:
        raise ValueError()

    b_input = b_input.to(device=config.device)
    b_target = b_target.to(device=config.device)

    return b_input.float(), b_target, grid_affines



def gaussian_likelihood(y_hat, log_var_scale, y_target):
    B,C,*_ = y_hat.shape
    scale = torch.exp(log_var_scale/2)
    dist = torch.distributions.Normal(y_hat, scale)

    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(y_target)

    # GLH
    return log_pxz



def kl_divergence(z, mean, std):
    # See https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    B,*_ = z.shape
    p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
    q = torch.distributions.Normal(mean, std)

    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # KL divergence
    kl = (log_qzx - log_pz)

    # Reduce spatial dimensions
    return kl.reshape(B,-1)



def get_ae_loss_value(y_hat, y_target, class_weights):
    return DC_and_CE_loss({}, {})(y_hat, y_target.argmax(1, keepdim=True))


def get_vae_loss_value(y_hat, y_target, z, mean, std, class_weights, model):
    recon_loss = get_ae_loss_value(y_hat, y_target, class_weights)#torch.nn.MSELoss()(y_hat, y_target)#gaussian_likelihood(y_hat, model.log_var_scale, y_target.float())
    # recon_loss = eo.reduce(recon_loss, 'B C spatial -> B ()', 'mean')
    kl = kl_divergence(z, mean, std)

    elbo = (0.1*kl + recon_loss).mean()

    return elbo

def model_step(config, epx, model, sa_atm, hla_atm, sa_cut_module, hla_cut_module, batch, label_tags, class_weights, segment_fn, autocast_enabled=False):

    ### Forward pass ###
    with amp.autocast(enabled=autocast_enabled):
        b_input, b_target, b_grid_affines = get_model_input(batch, config, len(label_tags), sa_atm, hla_atm, sa_cut_module, hla_cut_module, segment_fn)

        # nib.save(nib.Nifti1Image((
        #     b_target[0].argmax(0)
        #     + SkipConnector(mode='fill-sparse')(b_input, b_grid_affines)[0,6:].argmax(0).to(b_target)
        #     + SkipConnector(mode='fill-sparse')(b_input, b_grid_affines)[0,:6].argmax(0).to(b_target)
        # ).cpu().int().numpy(), affine=np.eye(4)), "out_sum_vol_slices.nii.gz")

        wanted_input_dim = 4 if 'hybrid' in config.model_type else 5
        assert b_input.dim() == wanted_input_dim, \
            f"Input image for model must be {wanted_input_dim}D but is {b_input.shape}"

        if config.model_type == 'vae':
            y_hat, (z, mean, std) = model(b_input)
        elif config.model_type in ['ae', 'hybrid-ae']:
            y_hat, _ = model(b_input)
        elif config.model_type in ['unet', 'unet-wo-skip', 'hybrid-unet-wo-skip']:
            y_hat = model(b_input)
        elif config.model_type == 'hybrid-unet':
            if 'stage-1' in wandb.run.name or 'stage-2' in wandb.run.name:
                b_grid_affines[0], b_grid_affines[1] = b_grid_affines[0].detach(), b_grid_affines[1].detach()
            y_hat = model(b_input, b_grid_affines)
        else:
            raise ValueError

        ### Calculate loss ###
        assert y_hat.dim() == 5, \
            f"Input shape for loss must be {5}D: BxNUM_CLASSESxSPATIAL but is {y_hat.shape}"
        assert b_target.dim() == 5, \
            f"Target shape for loss must be {5}D: BxNUM_CLASSESxSPATIAL but is {b_target.shape}"

        if "vae" in type(model).__name__.lower():
            loss = get_vae_loss_value(y_hat, b_target.float(), z, mean, std, class_weights, model)
        else:
            loss = get_ae_loss_value(y_hat, b_target.float(), class_weights)

    return y_hat, b_target, loss, b_input



def epoch_iter(epx, global_idx, config, model, sa_atm, hla_atm, sa_cut_module, hla_cut_module, dataset, dataloader, class_weights, phase='train',
    autocast_enabled=False, all_optimizers=None, scaler=None, store_net_output_to=None, r_params=None):
    PHASES = ['train', 'val', 'test']
    assert phase in ['train', 'val', 'test'], f"phase must be one of {PHASES}"

    epx_losses = []

    epx_sa_theta_aps = {}
    epx_hla_theta_aps = {}
    epx_sa_theta_zps = {}
    epx_hla_theta_zps = {}
    epx_sa_theta_t_offsets = {}
    epx_hla_theta_t_offsets = {}
    epx_input = {}

    label_scores_epoch = {}
    seg_metrics_nanmean = {}
    seg_metrics_std = {}
    seg_metrics_nanmean_oa = {}
    seg_metrics_std_oa = {}

    if phase == 'train':
        model.train()

        if config.train_affine_theta:
            if config.cuts_mode == 'sa>hla':
                sa_atm.eval()
                hla_atm.train()
            else:
                sa_atm.train()
                hla_atm.train()
        else:
            sa_atm.eval()
            hla_atm.eval()

        dataset.train(augment=config.do_augment)
    else:
        model.eval()
        sa_atm.eval()
        hla_atm.eval()
        dataset.eval()

    if isinstance(model, BlendowskiVAE):
        model.set_epoch(epx)

    segment_fn = dataset.segment_fn

    bbar = tqdm(enumerate(dataloader), desc=phase, total=len(dataloader))
    lst_mem = {}
    for batch_idx, batch in bbar:
        bbar.set_description(f"{phase}, {get_cuda_mem_info_str()}")
        if phase == 'train':
            for opt in all_optimizers.values():
                opt.zero_grad()

            y_hat, b_target, loss, b_input = model_step(
                config, epx,
                model, sa_atm, hla_atm, sa_cut_module, hla_cut_module,
                batch,
                dataset.label_tags, class_weights, segment_fn, autocast_enabled)

            if r_params is None:
                regularization = 0.0
            else:
                regularization = torch.cat([r().view(1,1).to(device=loss.device) for r in r_params.values()]).sum()

            loss = loss + regularization

            scaler.scale(loss).backward()
            # test_all_parameters_updated(model)
            # test_all_parameters_updated(sa_atm)
            # test_all_parameters_updated(hla_atm)
            for name, opt in all_optimizers.items():
                if name == 'transform_optimizer' and not config.train_affine_theta:
                    continue
                scaler.step(opt)
            scaler.update()

        else:
            with torch.no_grad():
                y_hat, b_target, loss, b_input = model_step(
                    config, epx,
                    model, sa_atm, hla_atm, sa_cut_module, hla_cut_module,
                    batch,
                    dataset.label_tags, class_weights, segment_fn, autocast_enabled)

        epx_losses.append(loss.item())

        epx_input.update({k:v for k,v in zip(batch['id'], b_input.cpu())})

        if sa_atm.last_theta_ap is not None:
            epx_sa_theta_aps.update({k:v for k,v in zip(batch['id'], sa_atm.last_theta_ap.cpu())})
        if sa_atm.last_theta_t_offsets is not None:
            epx_sa_theta_t_offsets.update({k:v for k,v in zip(batch['id'], sa_atm.last_theta_t_offsets.cpu())})
        if sa_atm.last_theta_zp is not None:
            epx_sa_theta_zps.update({k:v for k,v in zip(batch['id'], sa_atm.last_theta_zp.cpu())})
        if hla_atm.last_theta_ap is not None:
            epx_hla_theta_aps.update({k:v for k,v in zip(batch['id'], hla_atm.last_theta_ap.cpu())})
        if hla_atm.last_theta_t_offsets is not None:
            epx_hla_theta_t_offsets.update({k:v for k,v in zip(batch['id'], hla_atm.last_theta_t_offsets.cpu())})
        if hla_atm.last_theta_zp is not None:
            epx_hla_theta_zps.update({k:v for k,v in zip(batch['id'], hla_atm.last_theta_zp.cpu())})

        pred_seg = y_hat.argmax(1)

        # Load any dataloader sample affine matrix (all have been resampled the same spacing/orientation)
        nii_output_affine = batch['additional_data']['nifti_affine']
        # Taken from nibabel nifti1.py
        nifti_zooms = get_zooms(nii_output_affine).detach().cpu()
        # nifti_zooms = (nii_output_affine[:3,:3]*nii_output_affine[:3,:3]).sum(1).sqrt().detach().cpu()

        # Calculate fast dice score
        pred_seg_oh = eo.rearrange(torch.nn.functional.one_hot(pred_seg, len(training_dataset.label_tags)), 'b d h w oh -> b oh d h w')

        b_dice = monai.metrics.compute_dice(pred_seg_oh, b_target)

        label_scores_epoch = get_batch_score_per_label(label_scores_epoch, 'dice',
            b_dice, training_dataset.label_tags, exclude_bg=True)

        if (epx % 20 == 0 and epx > 0) or (epx+1 == config.epochs) or config.debug or config.test_only_and_output_to:
            b_sz = pred_seg_oh.shape[0]

            b_iou = monai.metrics.compute_iou(pred_seg_oh, b_target)
            label_scores_epoch = get_batch_score_per_label(label_scores_epoch, 'iou',
                b_iou, training_dataset.label_tags, exclude_bg=True)

            b_hd = monai.metrics.compute_hausdorff_distance(pred_seg_oh, b_target) * nifti_zooms.norm()
            b_hd = torch.cat([torch.zeros(b_sz,1).to(b_hd), b_hd], dim=1) # Add zero score for background
            label_scores_epoch = get_batch_score_per_label(label_scores_epoch, 'hd',
                b_hd, training_dataset.label_tags, exclude_bg=True)

            b_hd95 = monai.metrics.compute_hausdorff_distance(pred_seg_oh, b_target, percentile=95) * nifti_zooms.norm()
            b_hd95 = torch.cat([torch.zeros(b_sz,1).to(b_hd95), b_hd95], dim=1) # Add zero score for background
            label_scores_epoch = get_batch_score_per_label(label_scores_epoch, 'hd95',
                b_hd95, training_dataset.label_tags, exclude_bg=True)

            b_vol_ml = get_class_volumes(pred_seg, nifti_zooms, len(training_dataset.label_tags), unit='ml')
            b_vol_ml_target = get_class_volumes(b_target.argmax(1), nifti_zooms, len(training_dataset.label_tags), unit='ml')

            b_vol_diff = (b_vol_ml - b_vol_ml_target).abs()
            b_vol_rel_diff = (b_vol_diff / b_vol_ml_target)
            label_scores_epoch = get_batch_score_per_label(label_scores_epoch, 'delta_vol_ml',
                b_vol_diff, training_dataset.label_tags, exclude_bg=True)
            label_scores_epoch = get_batch_score_per_label(label_scores_epoch, 'delta_vol_rel',
                b_vol_rel_diff, training_dataset.label_tags, exclude_bg=True)

        if store_net_output_to not in ["", None]:
            store_path = Path(store_net_output_to, f"output_batch{batch_idx:05d}.pth")
            store_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(dict(batch=batch, input=b_input, output=y_hat, target=b_target), store_path)

        if config.debug: break

    (seg_metrics_nanmean,
     seg_metrics_std,
     seg_metrics_nanmean_oa,
     seg_metrics_std_oa) = reduce_label_scores_epoch(label_scores_epoch)

    loss_mean = torch.tensor(epx_losses).mean()
    ### Logging ###
    print(f"### {phase.upper()}")

    ### Log wandb data ###
    log_id = f'losses/{phase}_loss'
    log_val = loss_mean
    wandb.log({log_id: log_val}, step=global_idx)
    print(f'losses/{phase}_loss', log_val)

    log_label_metrics(f"scores/{phase}_mean", '', seg_metrics_nanmean, global_idx,
        logger_selected_metrics=('dice', 'iou', 'hd', 'hd95'), print_selected_metrics=('dice'))

    log_label_metrics(f"scores/{phase}_std", '', seg_metrics_std, global_idx,
        logger_selected_metrics=('dice', 'iou', 'hd', 'hd95'), print_selected_metrics=())

    log_oa_metrics(f"scores/{phase}_mean_oa_exclude_bg", '', seg_metrics_nanmean_oa, global_idx,
        logger_selected_metrics=('dice', 'iou', 'hd', 'hd95'), print_selected_metrics=('dice', 'iou', 'hd', 'hd95'))

    log_oa_metrics(f"scores/{phase}_std_oa_exclude_bg", '', seg_metrics_std_oa, global_idx,
        logger_selected_metrics=('dice', 'iou', 'hd', 'hd95'), print_selected_metrics=())

    print()
    output_dir = Path(f"data/output/{wandb.run.name}/{phase}")
    output_dir.mkdir(exist_ok=True, parents=True)

    mean_transform_dict = dict()

    if epx_sa_theta_aps:
        ornt_log_prefix = f"orientations/{phase}_sa_"
        sa_param_dict = dict(
            theta_ap=list(epx_sa_theta_aps.values()),
            theta_t_offsets=list(epx_sa_theta_t_offsets.values()),
            theta_zp=list(epx_sa_theta_zps.values()),
        )
        sa_theta_ap_mean, sa_theta_t_offsets_mean, sa_theta_zp_mean = \
            log_affine_param_stats(ornt_log_prefix, '', sa_param_dict, global_idx,
                logger_selected_metrics=('mean', 'std'), print_selected_metrics=('mean', 'std'))
        print()

        mean_transform_dict.update(
            dict(
                epoch_sa_theta_ap_mean=sa_theta_ap_mean,
                epoch_sa_theta_t_offsets_mean=sa_theta_t_offsets_mean,
                epoch_sa_theta_zp=sa_theta_zp_mean,
            )
        )

        if config.do_output:
            sa_dct = dict(
                epx_sa_theta_aps=epx_sa_theta_aps,
                epx_sa_theta_t_offsets=epx_sa_theta_t_offsets,
                epx_sa_theta_zps=epx_sa_theta_zps,
            )
            torch.save(sa_dct, output_dir/f"sa_params_{phase}_epx_{epx:05d}.pt")

    if epx_hla_theta_aps:
        ornt_log_prefix = f"orientations/{phase}_hla_"
        hla_param_dict = dict(
            theta_ap=list(epx_hla_theta_aps.values()),
            theta_t_offsets=list(epx_hla_theta_t_offsets.values()),
            theta_zp=list(epx_hla_theta_zps.values())
        )
        hla_theta_ap_mean, hla_theta_tp_mean, hla_theta_zp_mean = \
            log_affine_param_stats(ornt_log_prefix, '', hla_param_dict, global_idx,
                logger_selected_metrics=('mean', 'std'), print_selected_metrics=('mean', 'std'))
        print()

        mean_transform_dict.update(
            dict(
                epoch_hla_theta_ap_mean=hla_theta_ap_mean,
                epoch_hla_theta_tp_mean=hla_theta_tp_mean,
                epoch_hla_theta_zp=hla_theta_zp_mean,
            )
        )
        if config.do_output:
            hla_dct = dict(
                epx_hla_theta_aps=epx_hla_theta_aps,
                epx_hla_theta_t_offsets=epx_hla_theta_t_offsets,
                epx_hla_theta_zps=epx_hla_theta_zps,
            )
            torch.save(hla_dct, output_dir/f"hla_params_{phase}_epx_{epx:05d}.pt")

    if config.do_output and epx_input:
        # Store the slice model input
        save_input = torch.stack(list(epx_input.values()))

        if config.use_distance_map_localization:
            save_input =  (save_input < 0.5).float()

        if 'hybrid' in config.model_type:
            num_classes = len(training_dataset.label_tags)
            save_input = save_input.chunk(2,dim=1)
            save_input = torch.cat([slc.argmax(1, keepdim=True) for slc in save_input], dim=1)

        else:
            save_input = save_input.argmax(0)

        # Build mean image and concat to individual sample views
        save_input = torch.cat([save_input.float().mean(0, keepdim=True), save_input], dim=0)
        img_input = eo.rearrange(save_input, 'BI DI HI WI -> (DI WI) (BI HI)')
        log_frameless_image(img_input.numpy(), output_dir / f"slices_{phase}_epx_{epx:05d}.png", dpi=150, cmap='magma')

        lean_dct = {k:v for k,v in zip(epx_input.keys(), save_input.short())}
        torch.save(lean_dct, output_dir / f"input_{phase}_epx_{epx:05d}.pt")

    print(f"### END {phase.upper()}")
    print()
    print()

    return loss_mean, mean_transform_dict

def get_fold_postfix(fold_properties):
    fold_idx, _ = fold_properties
    return f'fold-{fold_idx}' if fold_idx != -1 else ""

def run_dl(run_name, config, fold_properties, stage=None, training_dataset=None, test_dataset=None):
    # reset_determinism()

    fold_idx, (train_idxs, val_idxs) = fold_properties

    best_quality_metric = 1.e16
    train_idxs = torch.tensor(train_idxs)
    train_ids = training_dataset.switch_3d_identifiers(train_idxs)
    val_idxs = torch.tensor(val_idxs)
    val_ids = training_dataset.switch_3d_identifiers(val_idxs)

    print(f"Will run training with these 3D samples (#{len(train_ids)}):", sorted(train_ids))
    print(f"Will run validation with these 3D samples (#{len(val_ids)}):", sorted(val_ids))

    ### Add train sampler and dataloaders ##
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idxs)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idxs)
    test_subsampler = torch.utils.data.SubsetRandomSampler(range(len(test_dataset)))

    if not run_test_once_only:
        train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size,
            sampler=train_subsampler, pin_memory=False, drop_last=True, # TODO Determine, why last batch is not transformed correctly
            collate_fn=training_dataset.get_efficient_augmentation_collate_fn()
        )
        training_dataset.set_augment_at_collate(False) # CAUTION: THIS INTERFERES WITH GRADIENT COMPUTATION IN AFFINE MODULES

        val_dataloader = DataLoader(training_dataset, batch_size=config.val_batch_size,
            sampler=val_subsampler, pin_memory=False, drop_last=False
        )

    test_dataloader = DataLoader(test_dataset, batch_size=config.val_batch_size,
        sampler=test_subsampler, pin_memory=False, drop_last=False
    )

    # Load from checkpoint, if any
    mdl_chk_path = config.model_checkpoint_path if 'model_checkpoint_path' in config else None
    (model, optimizer, scheduler, scaler), epx_start = get_model(
        config, len(training_dataset), len(training_dataset.label_tags),
        THIS_SCRIPT_DIR=THIS_SCRIPT_DIR, _path=mdl_chk_path, load_model_only=False,
        encoder_training_only=config.encoder_training_only)

    # Load transformation model from checkpoint, if any
    transform_mdl_chk_path = config.transform_model_checkpoint_path if 'transform_model_checkpoint_path' in config else None
    sa_atm_override = stage['sa_atm'] if stage is not None and 'sa_atm' in stage else None
    hla_atm_override = stage['hla_atm'] if stage is not None and 'hla_atm' in stage else None

    size_3d = training_dataset[0]['label'].shape[-3:] \
        if len(training_dataset) > 0 else test_dataset[0]['label'].shape[-3:]

    (sa_atm, hla_atm, sa_cut_module, hla_cut_module), transform_optimizer, transform_scheduler = get_transform_model(
        config, len(training_dataset.label_tags), size_3d, THIS_SCRIPT_DIR, _path=transform_mdl_chk_path,
        sa_atm_override=sa_atm_override, hla_atm_override=hla_atm_override)

    all_optimizers = dict(optimizer=optimizer, transform_optimizer=transform_optimizer)
    all_schedulers = dict(scheduler=scheduler, transform_scheduler=transform_scheduler)

    r_params = stage['r_params'] if stage is not None and 'r_params' in stage else None
    # all_bn_counts = torch.zeros([len(training_dataset.label_tags)], device='cpu')

    # for bn_counts in training_dataset.bincounts_3d.values():
    #     all_bn_counts += bn_counts

    # class_weights = 1 / (all_bn_counts).float().pow(.35)
    # class_weights /= class_weights.mean()

    # class_weights = class_weights.to(device=config.device)
    class_weights = None

    autocast_enabled = 'cuda' in config.device and config['use_autocast']

    for epx in range(epx_start, config.epochs):
        global_idx = get_global_idx(fold_idx, epx, config.epochs)
        # Log the epoch idx per fold - so we can recover the diagram by setting
        # ref_epoch_idx as x-axis in wandb interface
        print( f"### Log epoch {epx}/{config.epochs}")
        wandb.log({"ref_epoch_idx": epx}, step=global_idx)

        if not run_test_once_only:
            train_loss, mean_transform_dict = epoch_iter(
                epx, global_idx, config, model, sa_atm, hla_atm, sa_cut_module, hla_cut_module,
                training_dataset, train_dataloader, class_weights,
                phase='train', autocast_enabled=autocast_enabled,
                all_optimizers=all_optimizers, scaler=scaler, store_net_output_to=None,
                r_params=r_params)

            if stage:
                stage.update(mean_transform_dict)

            val_loss, _ = epoch_iter(epx, global_idx, config, model, sa_atm, hla_atm, sa_cut_module, hla_cut_module, training_dataset, val_dataloader, class_weights,
                phase='val', autocast_enabled=autocast_enabled, all_optimizers=None, scaler=None, store_net_output_to=None)

        test_loss, _ = epoch_iter(epx, global_idx, config, model, sa_atm, hla_atm, sa_cut_module, hla_cut_module, test_dataset, test_dataloader, class_weights,
            phase='test', autocast_enabled=autocast_enabled, all_optimizers=None, scaler=None, store_net_output_to=config.test_only_and_output_to)

        quality_metric = val_loss
        if run_test_once_only:
            break

        ###  Scheduler management ###
        if config.use_scheduling:
            for shd_name, shd in all_schedulers.items():
                if shd is not None:
                    shd.step()
                    wandb.log({f'training/{shd_name}_lr': shd.optimizer.param_groups[0]['lr']}, step=global_idx)
        print()

        # Save model
        if config.save_every is None:
            pass

        elif config.save_every == 'best':
            if quality_metric < best_quality_metric:
                best_quality_metric = quality_metric
                save_path = f"{config.mdl_save_prefix}/{wandb.run.name}_best"
                if stage is not None:
                    stage['save_path'] = save_path
                save_model(
                    Path(THIS_SCRIPT_DIR, save_path),
                    epx=epx,
                    loss=train_loss,
                    model=model,
                    sa_atm=sa_atm,
                    hla_atm=hla_atm,
                    sa_cut_module=sa_cut_module,
                    hla_cut_module=hla_cut_module,
                    optimizer=all_optimizers['optimizer'],
                    transform_optimizer=all_optimizers['transform_optimizer'],
                    scheduler=scheduler,
                    scaler=scaler)

        elif (epx % config.save_every == 0) or (epx+1 == config.epochs):
            save_path = f"{config.mdl_save_prefix}/{wandb.run.name}_epx{epx}"
            if stage is not None:
                stage['save_path'] = save_path
            save_model(
                Path(THIS_SCRIPT_DIR, save_path),
                epx=epx,
                loss=train_loss,
                model=model,
                sa_atm=sa_atm,
                hla_atm=hla_atm,
                sa_cut_module=sa_cut_module,
                hla_cut_module=hla_cut_module,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler)

            # (model, optimizer, scheduler, scaler) = \
            #     get_model(
            #         config, len(training_dataset),
            #         len(training_dataset.label_tags),
            #         THIS_SCRIPT_DIR=THIS_SCRIPT_DIR,
            #         _path=_path, device=config.device)

        # End of epoch loop

        if config.debug or run_test_once_only:
            break

# %%
# Config overrides
# config_dict['wandb_mode'] = 'disabled'
# config_dict['debug'] = True
# Model loading
# config_dict['checkpoint_path'] = 'ethereal-serenity-1138'
# config_dict['fold_override'] = 0

# Define sweep override dict
sweep_config_dict = dict(
    method='grid',
    metric=dict(goal='maximize', name='scores/val_dice_mean_left_atrium_fold0'),
    parameters=dict(
        # disturbance_mode=dict(
        #     values=[
        #        'LabelDisturbanceMode.AFFINE',
        #     ]
        # ),
        # disturbance_strength=dict(
        #     values=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        # ),
        # disturbed_percentage=dict(
        #     values=[0.3, 0.6]
        # ),
        # data_param_mode=dict(
        #     values=[
        #         DataParamMode.INSTANCE_PARAMS,
        #         DataParamMode.DISABLED,
        #     ]
        # ),
        use_risk_regularization=dict(
            values=[False, True]
        ),
        use_fixed_weighting=dict(
            values=[False, True]
        ),
        # fixed_weight_min_quantile=dict(
        #     values=[0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
        # ),
    )
)



# %%
def normal_run(config_dict, fold_properties, training_dataset, test_dataset):
    with wandb.init(project=PROJECT_NAME, group="training", job_type="train",
            config=config_dict, settings=wandb.Settings(start_method="thread"),
            mode=config_dict['wandb_mode']
        ) as run:
        run.name = f"{NOW_STR}_{run.name}_{get_fold_postfix(fold_properties)}"
        print("Running", run.name)
        config = wandb.config
        run_dl(run.name, config, fold_properties, training_dataset=training_dataset, test_dataset=test_dataset)



def stage_sweep_run(config_dict, fold_properties, all_stages, training_dataset, test_dataset):

    for stage in all_stages:
        stg_idx = all_stages.idx

        # Prepare stage settings
        stage.activate()

        stage_config = config_dict.copy()
        # Update intersecting keys of both
        stage_config.update((key, stage[key]) for key in set(stage).intersection(stage_config))
        print()

        with wandb.init(project=PROJECT_NAME, config=stage_config, settings=wandb.Settings(start_method="thread"),
            mode=stage_config['wandb_mode']) as run:

            run.name = f"{NOW_STR}_{run.name}_stage-{stg_idx+1}_{get_fold_postfix(fold_properties)}"
            print("Running", run.name)
            config = wandb.config

            run_dl(run.name, config, fold_properties, stage, training_dataset, test_dataset,)
        wandb.finish()
        torch.cuda.empty_cache()
        print(get_cuda_mem_info_str())



def wandb_sweep_run(config_dict, fold_properties, training_dataset, test_dataset):
    with wandb.init(
            settings=wandb.Settings(start_method="thread"),
            mode=config_dict['wandb_mode']) as run:

        run.name = f"{NOW_STR}_{run.name}_{get_fold_postfix(fold_properties)}"
        print("Running", run.name)
        config = wandb.config

        run_dl(run.name, config, fold_properties, training_dataset=training_dataset, test_dataset=test_dataset)



def clean_sweep_dict(config_dict):
    # Integrate all config_dict entries into sweep_dict.parameters -> sweep overrides config_dict
    cp_config_dict = copy.deepcopy(config_dict)

    for del_key in sweep_config_dict['parameters'].keys():
        if del_key in cp_config_dict:
            del cp_config_dict[del_key]
    merged_sweep_config_dict = copy.deepcopy(sweep_config_dict)

    for key, value in cp_config_dict.items():
        merged_sweep_config_dict['parameters'][key] = dict(value=value)

    # Convert enum values in parameters to string. They will be identified by their numerical index otherwise
    for key, param_dict in merged_sweep_config_dict['parameters'].items():
        if 'value' in param_dict and isinstance(param_dict['value'], Enum):
            param_dict['value'] = str(param_dict['value'])
        if 'values' in param_dict:
            param_dict['values'] = [str(elem) if isinstance(elem, Enum) else elem for elem in param_dict['values']]

        merged_sweep_config_dict['parameters'][key] = param_dict
    return merged_sweep_config_dict



def set_previous_stage_transform_chk(self):
    self['transform_model_checkpoint_path'] = self['save_path']



# main routine
#
#

# Configure folds
if config_dict.num_folds < 1:
    train_idxs = range(training_dataset.__len__(use_2d_override=False))
    val_idxs = []
    fold_idx = -1
    fold_iter = ([fold_idx, (train_idxs, val_idxs)],)

else:
    # kf = KFold(n_splits=config_dict.num_folds)
    # fold_iter = enumerate(kf.split(range(training_dataset.__len__(use_2d_override=False))))
    fold_iter = []
    for fold_idx in range(config_dict.num_folds):
        current_fold_idxs = training_dataset.data_split['train_folds'][f"fold_{fold_idx}"]
        train_files = [training_dataset.data_split['train_files'][idx] for idx in current_fold_idxs['train_idxs']]
        val_files = [training_dataset.data_split['train_files'][idx] for idx in current_fold_idxs['val_idxs']]

        train_ids = set([training_dataset.get_file_id(fl)[0] for fl in train_files])
        val_ids = set([training_dataset.get_file_id(fl)[0] for fl in val_files])
        assert len(train_ids.intersection(val_ids)) == 0, \
            f"Training and validation set must not overlap. But they do: {train_ids.intersection(val_ids)}"
        train_idxs = training_dataset.switch_3d_identifiers(train_ids)
        val_idxs = training_dataset.switch_3d_identifiers(val_ids)
        fold_iter.append((
            [idx for idx in train_idxs if idx is not None],
            [idx for idx in val_idxs if idx is not None]
        ))
    fold_iter = list(enumerate(fold_iter))

    if config_dict.get('fold_override', None):
        selected_fold = config_dict.get('fold_override', 0)
        fold_iter = fold_iter[selected_fold:selected_fold+1]


for fold_properties in fold_iter:
    if config_dict['sweep_type'] is None:
        normal_run(config_dict, fold_properties, training_dataset, test_dataset)

    elif config_dict['sweep_type'] == 'wandb_sweep':
        merged_sweep_config_dict = clean_sweep_dict(config_dict)
        sweep_id = wandb.sweep(merged_sweep_config_dict, project=PROJECT_NAME)

        def closure_wandb_sweep_run():
            return wandb_sweep_run(config_dict, fold_properties, training_dataset=training_dataset, test_dataset=test_dataset)

        wandb.agent(sweep_id, function=closure_wandb_sweep_run)

    elif config_dict['sweep_type'] == 'stage-sweep':

        size_3d = training_dataset[0]['label'].shape[-3:] \
            if len(training_dataset) > 0 else test_dataset[0]['label'].shape[-3:]
        r_params = init_regularization_params(
            [
                'hla_angles',
                'hla_offsets',
                'sa_angles',
                'sa_offsets',
            ], lambda_r=0.01)

        all_params_stages = [
            Stage( # Optimize SA
                r_params=r_params,
                use_random_affine_ap_init_sa=True,
                use_random_affine_ap_init_hla=True,
                sa_atm=get_atm(config_dict, len(training_dataset.label_tags), size_3d, 'sa', THIS_SCRIPT_DIR),
                # hla_atm=get_atm(config_dict, len(training_dataset.label_tags), size_3d, 'hla', THIS_SCRIPT_DIR),
                cuts_mode='sa',
                epochs=config_dict['epochs']*2,
                soft_cut_std=-999,
                do_augment=True,
                use_distance_map_localization=False,
                use_affine_theta=True,
                train_affine_theta=True,
                do_output=True,
                __activate_fn__=lambda self: None
            ),
            Stage( # Optimize hla
                # hla_atm=get_atm(config_dict, len(training_dataset.label_tags), size_3d, 'hla', THIS_SCRIPT_DIR),
                use_random_affine_ap_init_sa=False,
                use_random_affine_ap_init_hla=True,
                cuts_mode='sa>hla',
                epochs=config_dict['epochs']*2,
                soft_cut_std=-999,
                do_augment=True,
                use_distance_map_localization=False,
                use_affine_theta=True,
                train_affine_theta=True,
                do_output=True,
                __activate_fn__=set_previous_stage_transform_chk
            ),
            Stage( # Final optimized run
                use_random_affine_ap_init_sa=False,
                use_random_affine_ap_init_hla=False,
                do_output=True,
                cuts_mode='sa+hla',
                epochs=config_dict['epochs'],
                soft_cut_std=-999,
                do_augment=True,
                use_affine_theta=True,
                train_affine_theta=False,
                use_distance_map_localization=False,
                __activate_fn__=set_previous_stage_transform_chk
            ),
            Stage( # Reference run
                do_augment=True,
                do_output=True,
                cuts_mode='sa+hla',
                epochs=config_dict['epochs'],
                soft_cut_std=-999,
                train_affine_theta=False,
                use_affine_theta=False,
                use_distance_map_localization=False,
                __activate_fn__=lambda self: None
            ),
        ]

        selected_stages = all_params_stages
        stage_sweep_run(config_dict, fold_properties, StageIterator(selected_stages, verbose=True),
                        training_dataset=training_dataset, test_dataset=test_dataset)

    else:
        raise ValueError()

    if config_dict.debug or run_test_once_only:
        break
    # End of fold loop

# %%
if not in_notebook():
    sys.exit(0)

# %%
# Do any postprocessing / visualization in notebook here

# %%
