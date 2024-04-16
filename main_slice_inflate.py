import os
from pathlib import Path
import json
import dill
import einops as eo
from datetime import datetime
from git import Repo
import joblib
import argparse

import randomname

from slice_inflate.utils.common_utils import get_script_dir
THIS_SCRIPT_DIR = get_script_dir()

os.environ['CACHE_PATH'] = str(Path(THIS_SCRIPT_DIR, '.cache'))

from pytorch_run_on_recommended_gpu.run_on_recommended_gpu import get_cuda_environ_vars as get_vars
os.environ.update(get_vars(os.environ.get('MY_CUDA_VISIBLE_DEVICES','0')))

import torch
torch.set_printoptions(sci_mode=False)
# torch.autograd.set_detect_anomaly(True)
# import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb

import contextlib
from slice_inflate.utils.log_utils import get_global_idx, log_label_metrics, \
    log_oa_metrics, log_affine_param_stats, log_frameless_image, get_cuda_mem_info_str
from slice_inflate.datasets.clinical_cardiac_views import get_class_volumes
import numpy as np
import monai

from related_works.epix2vox.epix2vox import EPix2VoxModel128, get_optimizer_and_scheduler

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from slice_inflate.datasets.mmwhs_dataset import MMWHSDataset
from slice_inflate.datasets.mrxcat_dataset import MRXCATDataset

from slice_inflate.models.interface_models import NNUNET_InterfaceModel, EPix2Vox_InterfaceModel

from slice_inflate.utils.common_utils import DotDict
from slice_inflate.utils.torch_utils import get_batch_score_per_label, save_model, \
    reduce_label_scores_epoch, get_binarized_from_onehot_label

from slice_inflate.models.nnunet_models import Generic_UNet_Hybrid
from slice_inflate.models.learnable_transform import AffineTransformModule, HardCutModule
from slice_inflate.losses.regularization import init_regularization_params, Stage, StageIterator
from slice_inflate.utils.nifti_utils import get_zooms
from slice_inflate.utils.nifti_utils import nifti_grid_sample, get_zooms
from slice_inflate.models.learnable_transform import get_random_affine



def prepare_data(config):
    args = [config.dataset[1]]

    if config.dataset[0] == 'mmwhs':
        dataset_class = MMWHSDataset
    elif config.dataset[0] == 'mrxcat':
        dataset_class = MRXCATDataset
    else:
        raise ValueError()

    kwargs = {k:v for k,v in config.items()}

    cache_dir = 'git-' + config.git_commit.replace('!', '')
    cache_path = Path(os.environ['CACHE_PATH'], cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    arghash = joblib.hash(joblib.hash(args)+joblib.hash(kwargs))
    hashfile = cache_path / f"argshash_{arghash}_dataset.dil"
    if config.use_caching:
        if cache_path.is_file():
            print("Loading dataset from cache:", hashfile)
            with open(cache_path, 'rb') as file:
                dataset = dill.load(file)
        else:
            dataset = dataset_class(*args, **kwargs)
            print("Caching dataset:", hashfile)
            with open(hashfile, 'wb') as file:
                dill.dump(dataset, file)
    else:
        dataset = dataset_class(*args, **kwargs)

    return dataset



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


def get_model(config, dataset_len, num_classes, THIS_SCRIPT_DIR, _path=None, load_model_only=False):

    device = config.device
    assert config.model_type in ['hybrid-unet', 'hybrid-EPix2Vox', 'hybrid-Pix2Vox']
    if not _path is None:
        _path = Path(THIS_SCRIPT_DIR).joinpath(_path).resolve()

    if config.model_type == 'hybrid-unet':
        enc_mode = '2d'
        dec_mode = '3d'

        init_dict_path = Path(THIS_SCRIPT_DIR, "./slice_inflate/models/nnunet_init_dict_128_128_128.pkl")
        with open(init_dict_path, 'rb') as f:
            init_dict = dill.load(f)
        init_dict['num_classes'] = num_classes
        init_dict['deep_supervision'] = False
        init_dict['final_nonlin'] = torch.nn.Identity()
        init_dict['use_onehot_input'] = False
        init_dict['input_channels'] = num_classes*2
        init_dict['pool_op_kernel_sizes'][-1] = [2,2,2]
        init_dict['norm_op'] = nn.InstanceNorm3d
        # init_dict['convolutional_upsampling'] = True
        nnunet_model = Generic_UNet_Hybrid(**init_dict, use_skip_connections=True, encoder_mode=enc_mode, decoder_mode=dec_mode)

        seg_outputs = list(filter(lambda elem: 'seg_outputs' in elem[0], nnunet_model.named_parameters()))
        # Disable gradients of non-used deep supervision
        for so_idx in range(len(seg_outputs)-1):
            seg_outputs[so_idx][1].requires_grad = False
            model = NNUNET_InterfaceModel(nnunet_model)

    elif config.model_type == 'hybrid-EPix2Vox':
        epix_model = EPix2VoxModel128(
            use_merger=True, use_refiner=True, n_views=2, use_epix2vox=True,
            epoch_start_use_merger=0, epoch_start_use_refiner=0
        )
        model = EPix2Vox_InterfaceModel(epix_model)

    elif config.model_type == 'hybrid-Pix2Vox':
        epix_model = EPix2VoxModel128(
            use_merger=True, use_refiner=True, n_views=2, use_epix2vox=False,
            epoch_start_use_merger=0, epoch_start_use_refiner=0
        )
        model = EPix2Vox_InterfaceModel(epix_model)

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

    print(f"Trainable param count model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Non-trainable param count model: {sum(p.numel() for p in model.parameters() if not p.requires_grad)}")

    if config.model_type in ['hybrid-EPix2Vox', 'hybrid-Pix2Vox']:
        optimizer, scheduler = get_optimizer_and_scheduler(model.epix_model.encoder, model.epix_model.decoder, model.epix_model.merger, model.epix_model.refiner)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    scaler = amp.GradScaler()


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



def get_atm(config, num_classes, view, _path=None):
    assert view in ['sa', 'hla']
    device = config.device


    # Add atm models
    atm = AffineTransformModule(num_classes,
        torch.tensor(config.prescan_fov_mm),
        torch.tensor(config.prescan_fov_vox),
        torch.tensor(config.slice_fov_mm),
        torch.tensor(config.slice_fov_vox),

        offset_clip_value=config['offset_clip_value'],
        zoom_clip_value=config['zoom_clip_value'],
        optim_method=config.affine_theta_optim_method,
        tag=view,
        rotate_slice_to_min_principle=config.rotate_slice_to_min_principle)

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

def get_transform_model(config, num_classes, _path=None, sa_atm_override=None, hla_atm_override=None):
    device = config.device

    if isinstance(sa_atm_override, AffineTransformModule):
        # Check if atm is set externally
        sa_atm = sa_atm_override
    else:
        sa_atm = get_atm(config, num_classes, view='sa', _path=_path)

    if isinstance(hla_atm_override, AffineTransformModule):
        # Check if atm is set externally
        hla_atm = hla_atm_override
    else:
        hla_atm = get_atm(config, num_classes, view='hla', _path = _path)

    sa_atm.to(device)
    hla_atm.to(device)

    def set_requires_grad(module_list, requires_grad=True):
        for mod in module_list:
            for param in mod.parameters():
                param.requires_grad = requires_grad

    set_requires_grad([sa_atm, hla_atm], False)

    if config.cuts_mode == 'sa':
        set_requires_grad([sa_atm.localisation_net], True)
    elif config.cuts_mode in ['hla', 'sa>hla']:
        set_requires_grad([hla_atm.localisation_net], True)
    elif config.cuts_mode == 'sa+hla':
        set_requires_grad([
            sa_atm.localisation_net,
            hla_atm.localisation_net,
        ], True)
    else:
        raise ValueError()

    transform_parameters = (
        list(sa_atm.parameters())
        + list(hla_atm.parameters())
    )

    # else:
    #     raise ValueError()

    if config.train_affine_theta:
        assert config.use_affine_theta

    if config.train_affine_theta:
        transform_optimizer = torch.optim.AdamW(transform_parameters, weight_decay=0.1, lr=config.lr*2.0)
        transform_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(transform_optimizer, T_0=int(config.epochs/4)+1)
    else:
        transform_optimizer = NoneOptimizer()
        transform_scheduler = None

    if _path and config.train_affine_theta:
        assert Path(_path).is_dir()
        print(f"Loading transform optimizer from {_path}")
        transform_optimizer.load_state_dict(torch.load(Path(_path).joinpath('transform_optimizer.pth'), map_location=device))

    else:
        print(f"Generated fresh transform optimizer.")

    return (sa_atm, hla_atm), transform_optimizer, transform_scheduler



# %%
def get_transformed(config, phase, label, soft_label, nifti_affine, grid_affine_pre_mlp,
                    # hidden_augment_affine,
                    atm, image=None, segment_fn=None):

    img_is_invalid = image is None or image.dim() == 0

    B, num_classes, D, H, W = label.shape

    if img_is_invalid:
        image = torch.zeros(B,1,D,H,W, device=label.device)

    with amp.autocast(enabled=False):
        # Transform  label with 'bilinear' interpolation to have gradients
        soft_label_slc, label_slc, image_slc, grid_affine, atm_nii_affine = atm(
            soft_label.view(B, num_classes, D, H, W),
            label.view(B, num_classes, D, H, W),
            image.view(B, 1, D, H, W),
            nifti_affine, grid_affine_pre_mlp,
            # hidden_augment_affine
        )

    # nib.Nifti1Image(image[0][0].cpu().numpy(), affine=nifti_affine.cpu()[0].numpy()).to_filename('out_img_volume.nii.gz')
    # nib.Nifti1Image(label_slc[0].argmax(0).detach().int().cpu().numpy(), affine=atm_nii_affine[0].detach().cpu().numpy()).to_filename('out_lbl_slice.nii.gz')

    if config.label_slice_type == 'from-gt':
        pass
    elif config.label_slice_type == 'from-segmented' and phase != 'train':
        assert not img_is_invalid and segment_fn is not None
        with torch.no_grad():
            # Beware: Label slice does not have gradients anymore
            pred_slc = segment_fn(
                eo.rearrange(image_slc, "B C D H 1 -> B C 1 D H"),
                get_zooms(atm_nii_affine),
            ).long()
            if pred_slc.shape[-1] == 1:
                pass
            else:
                pred_slc = eo.rearrange(
                    pred_slc, "B 1 D H -> B D H 1"
                )  # MRXCAT segment output is 1,1,128,128, MMWHS output is 1,32,32,1
            soft_label_slc = label_slc = eo.rearrange(
                F.one_hot(pred_slc, num_classes), "B D H 1 OH -> B OH D H 1"
            ).to(soft_label_slc)
            # plt.imshow(image_slc[0].squeeze().cpu(), cmap='gray')
            # plt.imshow(label_slc[0].argmax(0).squeeze().cpu(), cmap='magma', alpha=.5, interpolation='none')
            # plt.savefig('slice_seg.png')

    if config.slice_fov_vox != config.hires_fov_vox:
        # Upsample label slice to hires resolution
        slice_target_shape = config.hires_fov_vox[:2] + [1]
        image_slc = F.interpolate(image_slc, size=slice_target_shape, mode='trilinear', align_corners=False)
        soft_label_slc = F.interpolate(soft_label_slc, size=slice_target_shape, mode='trilinear', align_corners=False)
        # label_slc = F.interpolate(label_slc.float(), size=slice_target_shape, mode='nearest').long()

    if img_is_invalid:
        image = torch.empty([])
        image_slc = torch.empty([])

    # Do not set label_slc to .long() here, since we (may) need the gradients
    return image_slc, soft_label_slc, grid_affine


def apply_affine_augmentation(affine_list, zoom_strength=0.1, offset_strength=0.1, rotation_strength=0.1):
    B = affine_list[0].shape[0]
    b_affine = []
    for _ in range(B):
        augment_affine = get_random_affine(
            rotation_strength=rotation_strength,
            zoom_strength=zoom_strength,
            offset_strength=offset_strength,
        )
        b_affine.append(augment_affine)
    b_affine = torch.stack(b_affine)

    for i in range(len(affine_list)):
        affine_list[i] = affine_list[i] @ b_affine.to(affine_list[i])

    return affine_list


def get_model_input(batch, phase, config, num_classes, sa_atm, hla_atm, segment_fn):
    b_label = batch['label']
    b_image = batch['image']

    # Get affines
    if config['clinical_view_affine_type'] == 'from-gt':
        b_view_affines = batch['additional_data']['gt_view_affines']
    elif config['clinical_view_affine_type'] == 'from-segmented':
        b_view_affines = batch['additional_data']['prescan_view_affines']
    nifti_affine = batch['additional_data']['nifti_affine']
    base_affine = torch.as_tensor(b_view_affines['centroids']).to(nifti_affine)

    # Transform volume to output space
    with torch.no_grad():
        b_label, _, nifti_affine = nifti_grid_sample(b_label.unsqueeze(1), nifti_affine,
            fov_mm=torch.as_tensor(config.hires_fov_mm), fov_vox=torch.as_tensor(config.hires_fov_vox),
            is_label=True, pre_grid_sample_affine=base_affine)
        b_image, _, _ = nifti_grid_sample(b_image.unsqueeze(1), nifti_affine,
            fov_mm=torch.as_tensor(config.hires_fov_mm), fov_vox=torch.as_tensor(config.hires_fov_vox),
            is_label=False, pre_grid_sample_affine=base_affine)
        b_label = b_label.squeeze(1)
        b_image = b_image.squeeze(1)

    b_label = eo.rearrange(F.one_hot(b_label, num_classes),
                        'B D H W OH -> B OH D H W')
    B,NUM_CLASSES,D,H,W = b_label.shape
    b_soft_label = b_label.float()

    sa_atm.use_affine_theta = config.use_affine_theta
    hla_atm.use_affine_theta = config.use_affine_theta

    if config.sa_view == 'RND':
        # Get a random view offset from prealingned volumes
        sa_input_grid_affine = sa_atm.random_grid_affine.repeat(B,1,1).to(nifti_affine)
    else:
        sa_input_grid_affine = base_affine.inverse() \
            @ torch.as_tensor(b_view_affines[config.sa_view]).view(B,4,4).to(nifti_affine)

    if config.hla_view == 'RND':
        # Get a random view offset from prealingned volumes
        hla_input_grid_affine = hla_atm.random_grid_affine.repeat(B,1,1).to(nifti_affine)
    else:
        hla_input_grid_affine = base_affine.inverse() \
            @ torch.as_tensor(b_view_affines[config.hla_view]).view(B,4,4).to(nifti_affine)

    if config.do_augment_input_orientation and phase in config.aug_phases:
        sa_input_grid_affine, hla_input_grid_affine = apply_affine_augmentation([sa_input_grid_affine, hla_input_grid_affine],
            rotation_strength=.1*config.sample_augment_strength,
            zoom_strength=.2*config.sample_augment_strength,
            offset_strength=0.,
        )


    if 'sa' in config.cuts_mode:
        sa_ctx = torch.no_grad if not sa_atm.training else contextlib.nullcontext
        with sa_ctx():
            sa_image_slc, sa_label_slc, sa_grid_affine = \
                get_transformed(
                    config,
                    phase,
                    b_label.view(B, NUM_CLASSES, D, H, W),
                    b_soft_label.view(B, NUM_CLASSES, D, H, W),
                    nifti_affine,
                    sa_input_grid_affine,
                    # hidden_augment_affine,
                    sa_atm,
                    image=b_image.view(B, 1, D, H, W), segment_fn=segment_fn)

            # Now apply augmentation that adds uncertainty to the inverse resconstruction grid sampling
            if config.do_augment_recon_orientation and phase in config.aug_phases:
                sa_grid_affine = apply_affine_augmentation([sa_grid_affine],
                    rotation_strength=.1*config.sample_augment_strength,
                    zoom_strength=.2*config.sample_augment_strength,
                    offset_strength=0.,
                )[0].to(nifti_affine)

            # from matplotlib import pyplot as plt
            # plt.imshow(sa_label_slc[0].argmax(0).squeeze().cpu().numpy())
            # plt.savefig('sa_label_slc.png')

    if 'hla' in config.cuts_mode:
        hla_ctx = torch.no_grad if not hla_atm.training else contextlib.nullcontext
        with hla_ctx():
            hla_image_slc, hla_label_slc, hla_grid_affine = \
                get_transformed(
                    config,
                    phase,
                    b_label.view(B, NUM_CLASSES, D, H, W),
                    b_soft_label.view(B, NUM_CLASSES, D, H, W),
                    nifti_affine,
                    hla_input_grid_affine,
                    # hidden_augment_affine,
                    hla_atm,
                    image=b_image.view(B, 1, D, H, W), segment_fn=segment_fn)

            # Now apply augmentation that adds uncertainty to the inverse resconstruction grid sampling
            if config.do_augment_recon_orientation and phase in config.aug_phases:
                hla_grid_affine = apply_affine_augmentation([hla_grid_affine],
                    rotation_strength=.1*config.sample_augment_strength,
                    zoom_strength=.2*config.sample_augment_strength,
                    offset_strength=0.,
                )[0].to(nifti_affine)

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

    if 'hybrid' in config.model_type:
        b_input = torch.cat(slices, dim=1).squeeze(-1)
        assert b_input.dim() == 4
    else:
        SPAT = config.hires_fov_vox[0]
        b_input = torch.cat(slices, dim=-1)
        b_input = torch.cat([b_input] * int(SPAT/b_input.shape[-1]), dim=-1) # Stack data hla/sa next to each other

    b_target = b_label

    # b_input = b_input.to(device=config.device)
    # b_target = b_target.to(device=config.device)

    return b_input, b_target, grid_affines



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



def get_loss_value(y_hat, y_target):
    return DC_and_CE_loss({}, {})(y_hat, y_target.argmax(1, keepdim=True))


def get_vae_loss_value(y_hat, y_target, z, mean, std, class_weights, model):
    recon_loss = get_loss_value(y_hat, y_target, class_weights)#torch.nn.MSELoss()(y_hat, y_target)#gaussian_likelihood(y_hat, model.log_var_scale, y_target.float())
    # recon_loss = eo.reduce(recon_loss, 'B C spatial -> B ()', 'mean')
    kl = kl_divergence(z, mean, std)

    elbo = (0.1*kl + recon_loss).mean()

    return elbo

def model_step(config, phase, epx, model, sa_atm, hla_atm, batch, label_tags, class_weights, segment_fn, autocast_enabled=False):

    ### Forward pass ###
    with amp.autocast(enabled=autocast_enabled):
        b_input, b_target, b_grid_affines = get_model_input(batch, phase, config, len(label_tags), sa_atm, hla_atm, segment_fn)

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
            y_hat = model(b_input, b_grid_affines)
        elif config.model_type in ['hybrid-EPix2Vox', 'hybrid-Pix2Vox']:
            assert config.use_binarized_labels == True, "(E)Pix2Vox only supports binary labels."
            y_hat = model(b_input, epx)
        else:
            raise ValueError

        torch.cuda.empty_cache()

        bg_lv_selector = torch.as_tensor(np.logical_or(
            np.array(label_tags) == 'LV',
            np.array(label_tags) == 'background')
        )

        if config.use_binarized_labels:
            b_target = get_binarized_from_onehot_label(b_target)
            y_hat = get_binarized_from_onehot_label(y_hat)

        ### Calculate loss ###
        assert y_hat.dim() == 5, \
            f"Input shape for loss must be {5}D: BxNUM_CLASSESxSPATIAL but is {y_hat.shape}"
        assert b_target.dim() == 5, \
            f"Target shape for loss must be {5}D: BxNUM_CLASSESxSPATIAL but is {b_target.shape}"

        if config.optimize_lv_only:
            loss = get_loss_value(y_hat[:,bg_lv_selector], b_target[:,bg_lv_selector])
        else:
            loss = get_loss_value(y_hat, b_target)

    return y_hat, b_target, loss, b_input



def epoch_iter(epx, global_idx, config, model, sa_atm, hla_atm, dataset, dataloader, class_weights, phase='train',
    autocast_enabled=False, all_optimizers=None, scaler=None, store_net_output_to=None, r_params=None):
    PHASES = ['train', 'val', 'test']
    assert phase in ['train', 'val', 'test'], f"phase must be one of {PHASES}"

    epx_losses = []

    epx_sa_theta_aps = {}
    epx_sa_theta_zps = {}
    epx_sa_theta_t_offsets = {}
    epx_sa_theta_grid_affines = {}
    epx_sa_transformed_nii_affines = {}

    epx_hla_theta_aps = {}
    epx_hla_theta_zps = {}
    epx_hla_theta_t_offsets = {}
    epx_hla_theta_grid_affines = {}
    epx_hla_transformed_nii_affines = {}

    epx_input = {}

    label_scores_epoch = {}
    seg_metrics_nanmean = {}
    seg_metrics_std = {}
    seg_metrics_nanmean_oa = {}
    seg_metrics_std_oa = {}

    if phase == 'train':
        model.train()

        if config.train_affine_theta:
            if 'sa' in config.cuts_mode and not config.cuts_mode == 'sa>hla':
                sa_atm.train()
            else:
                sa_atm.eval()

            if 'hla' in config.cuts_mode:
                hla_atm.train()
            else:
                hla_atm.eval()
        else:
            sa_atm.eval()
            hla_atm.eval()

    else:
        model.eval()
        sa_atm.eval()
        hla_atm.eval()
        dataset.eval()

    segment_fn = dataset.segment_fn

    bbar = tqdm(enumerate(dataloader), desc=phase, total=len(dataloader))
    lst_mem = {}
    if config.use_binarized_labels:
        eval_label_tags = ("background", "foreground")
    else:
        eval_label_tags = dataset.label_tags

    for batch_idx, batch in bbar:
        bbar.set_description(f"{phase}, {get_cuda_mem_info_str()}")
        if phase == 'train':
            y_hat, b_target, loss, b_input = model_step(
                config, phase, epx,
                model, sa_atm, hla_atm,
                batch,
                dataset.label_tags, class_weights, segment_fn, autocast_enabled)

            if r_params is None:
                regularization = 0.0
            else:
                regularization = torch.cat([r().view(1,1).to(device=loss.device) for r in r_params.values()]).sum()

            loss = loss + regularization
            loss_accum = loss / config.num_grad_accum_steps

            if config.use_autocast:
                scaler.scale(loss_accum).backward()
            else:
                loss_accum.backward()

            if (batch_idx+1) % config.num_grad_accum_steps != 0:
                continue

            if config.use_autocast:
                for name, opt in all_optimizers.items():
                    if name == 'transform_optimizer' and not config.train_affine_theta:
                        continue
                    scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            else:
                for name, opt in all_optimizers.items():
                    if name == 'transform_optimizer' and not config.train_affine_theta:
                        continue
                    opt.step()
                    opt.zero_grad()

            epx_losses.append((loss_accum*config.num_grad_accum_steps).item())

        else:
            with torch.no_grad():
                y_hat, b_target, loss, b_input = model_step(
                    config, phase, epx,
                    model, sa_atm, hla_atm,
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
        if sa_atm.last_grid_affine is not None:
            epx_sa_theta_grid_affines.update({k:v for k,v in zip(batch['id'], sa_atm.last_grid_affine.cpu())})
        if sa_atm.last_transformed_nii_affine is not None:
            epx_sa_transformed_nii_affines.update({k:v for k,v in zip(batch['id'], sa_atm.last_transformed_nii_affine.cpu())})

        if hla_atm.last_theta_ap is not None:
            epx_hla_theta_aps.update({k:v for k,v in zip(batch['id'], hla_atm.last_theta_ap.cpu())})
        if hla_atm.last_theta_t_offsets is not None:
            epx_hla_theta_t_offsets.update({k:v for k,v in zip(batch['id'], hla_atm.last_theta_t_offsets.cpu())})
        if hla_atm.last_theta_zp is not None:
            epx_hla_theta_zps.update({k:v for k,v in zip(batch['id'], hla_atm.last_theta_zp.cpu())})
        if hla_atm.last_grid_affine is not None:
            epx_hla_theta_grid_affines.update({k:v for k,v in zip(batch['id'], hla_atm.last_grid_affine.cpu())})
        if hla_atm.last_transformed_nii_affine is not None:
            epx_hla_transformed_nii_affines.update({k:v for k,v in zip(batch['id'], hla_atm.last_transformed_nii_affine.cpu())})

        pred_seg = y_hat.argmax(1)

        # Load any dataloader sample affine matrix (all have been resampled the same spacing/orientation)
        nii_output_affine = batch['additional_data']['nifti_affine']
        # Taken from nibabel nifti1.py
        nifti_zooms = get_zooms(nii_output_affine).detach().cpu()
        # nifti_zooms = (nii_output_affine[:3,:3]*nii_output_affine[:3,:3]).sum(1).sqrt().detach().cpu()

        # Calculate fast dice score
        pred_seg_oh = eo.rearrange(torch.nn.functional.one_hot(pred_seg, len(eval_label_tags)), 'b d h w oh -> b oh d h w')

        b_dice = monai.metrics.compute_dice(pred_seg_oh, b_target)

        label_scores_epoch = get_batch_score_per_label(label_scores_epoch, 'dice',
            b_dice, eval_label_tags, exclude_bg=True)

        if (epx % 20 == 0 and epx > 0) or (epx+1 == config.epochs) or config.debug or config.test_only_and_output_to:
            b_sz = pred_seg_oh.shape[0]

            b_iou = monai.metrics.compute_iou(pred_seg_oh, b_target)
            label_scores_epoch = get_batch_score_per_label(label_scores_epoch, 'iou',
                b_iou, eval_label_tags, exclude_bg=True)

            b_hd = monai.metrics.compute_hausdorff_distance(pred_seg_oh, b_target) * nifti_zooms.norm()
            b_hd = torch.cat([torch.zeros(b_sz,1).to(b_hd), b_hd], dim=1) # Add zero score for background
            label_scores_epoch = get_batch_score_per_label(label_scores_epoch, 'hd',
                b_hd, eval_label_tags, exclude_bg=True)

            b_hd95 = monai.metrics.compute_hausdorff_distance(pred_seg_oh, b_target, percentile=95) * nifti_zooms.norm()
            b_hd95 = torch.cat([torch.zeros(b_sz,1).to(b_hd95), b_hd95], dim=1) # Add zero score for background
            label_scores_epoch = get_batch_score_per_label(label_scores_epoch, 'hd95',
                b_hd95, eval_label_tags, exclude_bg=True)

            b_vol_ml = get_class_volumes(pred_seg, nifti_zooms, len(eval_label_tags), unit='ml')
            b_vol_ml_target = get_class_volumes(b_target.argmax(1), nifti_zooms, len(eval_label_tags), unit='ml')

            b_vol_diff = (b_vol_ml - b_vol_ml_target).abs()
            b_vol_rel_diff = (b_vol_diff / b_vol_ml_target)
            label_scores_epoch = get_batch_score_per_label(label_scores_epoch, 'delta_vol_ml',
                b_vol_diff, eval_label_tags, exclude_bg=True)
            label_scores_epoch = get_batch_score_per_label(label_scores_epoch, 'delta_vol_rel',
                b_vol_rel_diff, eval_label_tags, exclude_bg=True)

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
        logger_selected_metrics=('dice', 'iou', 'hd', 'hd95', 'delta_vol_ml', 'delta_vol_rel'), print_selected_metrics=())

    log_label_metrics(f"scores/{phase}_std", '', seg_metrics_std, global_idx,
        logger_selected_metrics=('dice', 'iou', 'hd', 'hd95', 'delta_vol_ml', 'delta_vol_rel'), print_selected_metrics=())

    log_oa_metrics(f"scores/{phase}_mean_oa_exclude_bg", '', seg_metrics_nanmean_oa, global_idx,
        logger_selected_metrics=('dice', 'iou', 'hd', 'hd95', 'delta_vol_ml', 'delta_vol_rel'), print_selected_metrics=('dice', 'hd95'))

    log_oa_metrics(f"scores/{phase}_std_oa_exclude_bg", '', seg_metrics_std_oa, global_idx,
        logger_selected_metrics=('dice', 'iou', 'hd', 'hd95', 'delta_vol_ml', 'delta_vol_rel'), print_selected_metrics=())

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
                logger_selected_metrics=('mean', 'std'), print_selected_metrics=())
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
                epx_sa_theta_grid_affines=epx_sa_theta_grid_affines,
                epx_sa_transformed_nii_affines=epx_sa_transformed_nii_affines,
            )
            torch.save(sa_dct, output_dir/f"sa_params_{phase}_epx_{epx:05d}.pth")

    if epx_hla_theta_aps:
        ornt_log_prefix = f"orientations/{phase}_hla_"
        hla_param_dict = dict(
            theta_ap=list(epx_hla_theta_aps.values()),
            theta_t_offsets=list(epx_hla_theta_t_offsets.values()),
            theta_zp=list(epx_hla_theta_zps.values())
        )
        hla_theta_ap_mean, hla_theta_tp_mean, hla_theta_zp_mean = \
            log_affine_param_stats(ornt_log_prefix, '', hla_param_dict, global_idx,
                logger_selected_metrics=('mean', 'std'), print_selected_metrics=())
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
                epx_hla_theta_grid_affines=epx_hla_theta_grid_affines,
                epx_hla_transformed_nii_affines=epx_hla_transformed_nii_affines,
            )
            torch.save(hla_dct, output_dir/f"hla_params_{phase}_epx_{epx:05d}.pt")

    if config.do_output and epx_input:
        # Store the slice model input
        save_input = torch.stack(list(epx_input.values()))

        if 'hybrid' in config.model_type:
            num_classes = len(eval_label_tags)
            save_input = save_input.chunk(2,dim=1)
            save_input = torch.cat([slc.argmax(1, keepdim=True) for slc in save_input], dim=1)

        else:
            save_input = save_input.argmax(0)

        # Build mean image and concat to individual sample views
        save_input = torch.cat([save_input.float().mean(0, keepdim=True), save_input], dim=0)
        img_input = eo.rearrange(save_input, 'BI DI HI WI -> (DI WI) (BI HI)')
        img_input[img_input == 0] = np.nan
        log_frameless_image(img_input.numpy(), output_dir / f"slices_{phase}_epx_{epx:05d}.png", dpi=150, cmap='RdPu')

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

    training_dataset.set_segment_fn(fold_idx)
    test_dataset.set_segment_fn(fold_idx)

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
            sampler=train_subsampler, pin_memory=False, drop_last=True
        )

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
        THIS_SCRIPT_DIR=THIS_SCRIPT_DIR, _path=mdl_chk_path, load_model_only=False)

    # Load transformation model from checkpoint, if any
    transform_mdl_chk_path = config.transform_model_checkpoint_path if 'transform_model_checkpoint_path' in config else None
    sa_atm_override = stage['sa_atm'] if stage is not None and 'sa_atm' in stage else None
    hla_atm_override = stage['hla_atm'] if stage is not None and 'hla_atm' in stage else None

    (sa_atm, hla_atm), transform_optimizer, transform_scheduler = get_transform_model(
        config, len(training_dataset.label_tags), _path=transform_mdl_chk_path,
        sa_atm_override=sa_atm_override, hla_atm_override=hla_atm_override)

    all_optimizers = dict(optimizer=optimizer, transform_optimizer=transform_optimizer)
    all_schedulers = dict(scheduler=scheduler, transform_scheduler=transform_scheduler)

    r_params = stage['r_params'] if stage is not None and 'r_params' in stage else None
    class_weights = None

    autocast_enabled = 'cuda' in config.device and config['use_autocast']

    # c_model = torch.compile(model)
    # c_sa_atm = torch.compile(sa_atm)
    # c_hla_atm = torch.compile(hla_atm)

    for epx in range(epx_start, config.epochs):
        global_idx = get_global_idx(fold_idx, epx, config.epochs)
        # Log the epoch idx per fold - so we can recover the diagram by setting
        # ref_epoch_idx as x-axis in wandb interface
        print( f"### Log epoch {epx}/{config.epochs}")
        wandb.log({"ref_epoch_idx": epx}, step=global_idx)

        if not run_test_once_only:
            train_loss, mean_transform_dict = epoch_iter(
                epx, global_idx, config, model, sa_atm, hla_atm,
                training_dataset, train_dataloader, class_weights,
                phase='train', autocast_enabled=autocast_enabled,
                all_optimizers=all_optimizers, scaler=scaler, store_net_output_to=None,
                r_params=r_params)

            if stage:
                stage.update(mean_transform_dict)

            val_loss, _ = epoch_iter(epx, global_idx, config, model, sa_atm, hla_atm, training_dataset, val_dataloader, class_weights,
                phase='val', autocast_enabled=autocast_enabled, all_optimizers=None, scaler=None, store_net_output_to=None)

        test_loss, _ = epoch_iter(epx, global_idx, config, model, sa_atm, hla_atm, test_dataset, test_dataloader, class_weights,
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



def normal_run(run_name, config_dict, fold_properties, training_dataset, test_dataset):

    with wandb.init(project=PROJECT_NAME, group="training", job_type="train",
            config=config_dict, settings=wandb.Settings(start_method="thread"),
            mode=config_dict['wandb_mode']
        ) as run:
        run.name = run_name
        print("Running", run.name)
        config = wandb.config
        run_dl(run.name, config, fold_properties, training_dataset=training_dataset, test_dataset=test_dataset)



def stage_sweep_run(run_name, config_dict, fold_properties, all_stages, training_dataset, test_dataset):
    for stage in all_stages:
        stg_id = all_stages.current_key

        # Prepare stage settings
        stage.activate()

        stage_config = config_dict.copy()
        # Update intersecting keys of both
        stage_config.update((key, stage[key]) for key in set(stage).intersection(stage_config))
        print()

        torch.cuda.empty_cache()
        with wandb.init(project=PROJECT_NAME, config=stage_config, settings=wandb.Settings(start_method="thread"),
            mode=stage_config['wandb_mode']) as run:

            run.name = f"{run_name}_stage-{stg_id}"
            print("Running", run.name)
            config = wandb.config

            run_dl(run.name, config, fold_properties, stage, training_dataset, test_dataset,)
        wandb.finish()
        torch.cuda.empty_cache()
        print(get_cuda_mem_info_str())



def set_previous_stage_transform_chk(self):
    self['transform_model_checkpoint_path'] = self['save_path']


if __name__ == '__main__':
    # Add argument parser for additional config file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_config_path', type=str, default=None, help='Path to config file')
    args = parser.parse_args()

    if args.meta_config_path is not None:
        with open(args.meta_config_path, 'r') as f:
            meta_config_dict = DotDict(json.load(f))
    else:
        meta_config_dict = DotDict()

    NOW_STR = datetime.now().strftime("%Y%m%d__%H_%M_%S")
    THIS_REPO = Repo(THIS_SCRIPT_DIR)
    PROJECT_NAME = "slice_inflate"

    with open(Path(THIS_SCRIPT_DIR, 'config_dict.json'), 'r') as f:
        config_dict = DotDict(json.load(f))

    # Merge meta config
    config_dict.update(meta_config_dict)

    # Log commmit id and dirtiness
    dirty_str = "!dirty-" if THIS_REPO.is_dirty() else ""
    config_dict['git_commit'] = f"{dirty_str}{THIS_REPO.commit().hexsha}"

    run_test_once_only = not (config_dict.test_only_and_output_to in ["", None])

    train_config = DotDict(config_dict.copy())
    if run_test_once_only:
        train_config['state'] = 'empty'
    training_dataset = prepare_data(train_config)

    test_config = DotDict(config_dict.copy())
    test_config['state'] = 'test'
    test_dataset = prepare_data(test_config)

    # Configure folds
    if config_dict.num_folds < 1:
        train_idxs = range(len(training_dataset))
        val_idxs = []
        fold_idx = -1
        fold_iter = ([fold_idx, (train_idxs, val_idxs)],)

    else:
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

        if config_dict['fold_override'] is not None:
            selected_fold = config_dict['fold_override']
            fold_iter = fold_iter[selected_fold:selected_fold+1]

    rnd_name = randomname.get_name()
    run_name = f"{NOW_STR}_{rnd_name}"

    for fold_properties in fold_iter:
        run_name_with_fold = run_name + f"_{get_fold_postfix(fold_properties)}"
        if config_dict['sweep_type'] is None:
            normal_run(run_name_with_fold, config_dict, fold_properties, training_dataset, test_dataset)

        elif config_dict['sweep_type'] == 'stage-sweep':

            r_params = init_regularization_params(
                [
                    'hla_angles',
                    'hla_offsets',
                    'sa_angles',
                    'sa_offsets',
                ], lambda_r=0.01)

            all_params_stages = dict(
                opt_first=Stage( # Optimize SA
                    r_params=r_params,
                    cuts_mode='sa',
                    epochs=int(config_dict['epochs']*1.0),
                    use_affine_theta=True,
                    train_affine_theta=True,
                    do_output=True,
                    __activate_fn__=lambda self: None
                ),
                opt_second=Stage( # Optimize hla
                    r_params=r_params,
                    cuts_mode='sa>hla',
                    epochs=int(config_dict['epochs']*1.0),
                    use_affine_theta=True,
                    train_affine_theta=True,
                    do_output=True,
                    __activate_fn__=set_previous_stage_transform_chk
                ),
                opt_both_fix=Stage( # Final optimized run
                    do_output=True,
                    cuts_mode='sa+hla',
                    epochs=config_dict['epochs'],
                    use_affine_theta=True,
                    train_affine_theta=False,
                    __activate_fn__=set_previous_stage_transform_chk
                ),
                ref=Stage( # Reference run
                    do_output=True,
                    cuts_mode='sa+hla',
                    epochs=config_dict['epochs'],
                    train_affine_theta=False,
                    use_affine_theta=False,
                    __activate_fn__=lambda self: None
                ),
            )

            selected_stages = all_params_stages

            if 'stage_override' in config_dict and config_dict['stage_override'] is not None:
                selected_stages = {k:v for k,v in all_params_stages.items() if config_dict['stage_override'] == k}
            stage_iterator = StageIterator(selected_stages, verbose=True)

            stage_sweep_run(run_name_with_fold, config_dict, fold_properties, stage_iterator,
                            training_dataset=training_dataset, test_dataset=test_dataset)

        else:
            raise ValueError()

        if config_dict.debug or run_test_once_only:
            break
        # End of fold loop