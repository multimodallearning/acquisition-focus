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
from pathlib import Path
import json
import dill
import einops as eo
from datetime import datetime

os.environ['MMWHS_CACHE_PATH'] = str(Path('.', '.cache'))

from meidic_vtach_utils.run_on_recommended_cuda import get_cuda_environ_vars as get_vars
os.environ.update(get_vars('*'))

import torch
torch.set_printoptions(sci_mode=False)
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb
import nibabel as nib

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from slice_inflate.datasets.align_mmwhs import cut_slice
from slice_inflate.utils.log_utils import get_global_idx, log_label_metrics, log_oa_metrics
from sklearn.model_selection import KFold
import numpy as np
from mdl_seg_class.metrics import dice3d, hausdorff3d

from slice_inflate.losses.dice_loss import DC_and_CE_loss
from slice_inflate.datasets.mmwhs_dataset import MMWHSDataset, load_data, extract_2d_data
from slice_inflate.utils.common_utils import DotDict, get_script_dir, in_notebook
from slice_inflate.utils.torch_utils import reset_determinism, ensure_dense, \
    get_batch_dice_over_all, get_batch_score_per_label, save_model, \
    reduce_label_scores_epoch, get_test_func_all_parameters_updated, anomaly_hook
from slice_inflate.models.nnunet_models import Generic_UNet_Hybrid
from slice_inflate.models.affine_transform import AffineTransformModule, get_random_angles, SoftCutModule, HardCutModule, get_theta_params
from slice_inflate.models.ae_models import BlendowskiAE, BlendowskiVAE, HybridAE
from slice_inflate.losses.regularization import optimize_sa_angles, optimize_sa_offsets, optimize_hla_angles, optimize_hla_offsets, init_regularization_params, Stage, StageIterator

NOW_STR = datetime.now().strftime("%Y%d%m__%H_%M_%S")
THIS_SCRIPT_DIR = get_script_dir()
PROJECT_NAME = "slice_inflate"

training_dataset, test_dataset = None, None
test_all_parameters_updated = get_test_func_all_parameters_updated()
# %%

with open(Path(THIS_SCRIPT_DIR, 'config_dict.json'), 'r') as f:
    config_dict = DotDict(json.load(f))

def prepare_data(config):
    training_dataset = MMWHSDataset(
        config.data_base_path,
        state=config.state,
        load_func=load_data,
        extract_slice_func=extract_2d_data,
        modality=config.modality,
        do_align_global=True,
        do_resample=False, # Prior to cropping, resample image?
        crop_3d_region=None, # Crop or pad the images to these dimensions
        fov_mm=config.fov_mm,
        fov_vox=config.fov_vox,
        crop_around_3d_label_center=config.crop_around_3d_label_center,
        pre_interpolation_factor=1., # When getting the data, resize the data by this factor
        ensure_labeled_pairs=True, # Only use fully labelled images (segmentation label available)
        use_2d_normal_to=config.use_2d_normal_to, # Use 2D slices cut normal to D,H,>W< dimensions
        crop_around_2d_label_center=config.crop_around_2d_label_center,
        max_load_3d_num=config.max_load_3d_num,
        soft_cut_std=config.soft_cut_std,
        augment_angle_std=5,

        device=config.device,
        debug=config.debug
    )

    return training_dataset


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
if False:
    training_dataset.train(augment=True)
    training_dataset.self_attributes['augment_angle_std'] = 1
    print("do_augment", training_dataset.do_augment)
    for sample in [training_dataset[idx] for idx in range(20)]:
        fig = plt.figure(figsize=(16., 1.))

        show_row = [
            # cut_slice(sample['image']),
            cut_slice(sample['label'].unsqueeze(0)).argmax(1).squeeze(),

            # sample['sa_image_slc'],
            sample['sa_label_slc'].unsqueeze(0).argmax(1).squeeze(),

            # sample['hla_image_slc'],
            sample['hla_label_slc'].unsqueeze(0).argmax(1).squeeze(),
        ]

        show_row = [sh.cpu() for sh in show_row]

        grid = ImageGrid(fig, 111,  # similar to subplot(111)
            nrows_ncols=(1, len(show_row)),  # creates 2x2 grid of axes
            axes_pad=0.0,  # pad between axes in inch.
        )

        for ax, im in zip(grid, show_row):
            ax.imshow(im, cmap='gray', interpolation='none')

        plt.show()

# %%
if False:
    training_dataset.train(augment=True)
    training_dataset.self_attributes['augment_angle_std'] = 1
    print("do_augment", training_dataset.do_augment)

    train_dataloader = DataLoader(training_dataset, batch_size=config_dict.batch_size,
        pin_memory=False, drop_last=False,
        collate_fn=training_dataset.get_efficient_augmentation_collate_fn()
    )
    training_dataset.set_augment_at_collate(False)

    for batch in train_dataloader:
        fig = plt.figure(figsize=(16., 1.))

        show_row = \
            [sh for sh in cut_slice(batch['label']).argmax(1).squeeze()] + \
            [sh for sh in batch['sa_label_slc'].argmax(1).squeeze()] + \
            [sh for sh in batch['hla_label_slc'].argmax(1).squeeze()]

        show_row = [sh.cpu() for sh in show_row]

        grid = ImageGrid(fig, 111,  # similar to subplot(111)
            nrows_ncols=(1, len(show_row)),  # creates 2x2 grid of axes
            axes_pad=0.0,  # pad between axes in inch.
        )

        for ax, im in zip(grid, show_row):
            ax.imshow(im, cmap='gray', interpolation='none')

        plt.show()

# %%
if False:
    training_dataset.train()

    training_dataset.self_attributes['augment_angle_std'] = 5
    print("do_augment", training_dataset.do_augment)
    for sample_idx in range(20):
        lbl, sa_label, hla_label = torch.zeros(128,128), torch.zeros(128,128), torch.zeros(128,128)
        for augment_idx in range(15):
            sample = training_dataset[sample_idx]
            nib.save(nib.Nifti1Image(sample['label'].cpu().numpy(), affine=torch.eye(4).numpy()), f'out{sample_idx}.nii.gz')
            lbl += cut_slice(sample['label']).cpu()
            sa_label += sample['sa_label_slc'].cpu()
            hla_label += sample['hla_label_slc'].cpu()

        fig = plt.figure(figsize=(16., 4.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
            nrows_ncols=(1, 3),  # creates 2x2 grid of axes
            axes_pad=0.0,  # pad between axes in inch.
        )

        show_row = [
            lbl, sa_label, hla_label
        ]

        for ax, im in zip(grid, show_row):
            ax.imshow(im, cmap='magma', interpolation='none')

        plt.show()

# %%
if False:
    training_dataset.train(augment=False)
    training_dataset.self_attributes['augment_angle_std'] = 2
    print(training_dataset.do_augment)

    lbl, sa_label, hla_label = torch.zeros(128,128), torch.zeros(128,128), torch.zeros(128,128)
    for tr_idx in range(len(training_dataset)):
        sample = training_dataset[tr_idx]

        lbl += cut_slice(sample['label']).cpu()
        sa_label += sample['sa_label_slc'].cpu()
        hla_label += sample['hla_label_slc'].cpu()

    fig = plt.figure(figsize=(16., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
        nrows_ncols=(1, 3),  # creates 2x2 grid of axes
        axes_pad=0.0,  # pad between axes in inch.
    )

    show_row = [
        lbl, sa_label, hla_label
    ]

    for ax, im in zip(grid, show_row):
        ax.imshow(im, cmap='magma', interpolation='none')

    plt.show()


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
    assert config.model_type in ['vae', 'ae', 'hybrid-ae', 'unet', 'unet-wo-skip', 'hybrid-unet-wo-skip']
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

            def forward(self, x):
                y_hat = self.nnunet_model(x)
                if isinstance(y_hat, tuple):
                    return y_hat[0], None
                else:
                    return y_hat, None

        model = InterfaceModel(nnunet_model)

    else:
        raise ValueError

    model.to(device)


    if config['soft_cut_std'] > 0:
        sa_cut_module = SoftCutModule(soft_cut_softness=config['soft_cut_std'])
        hla_cut_module = SoftCutModule(soft_cut_softness=config['soft_cut_std'])
    else:
        sa_cut_module = HardCutModule()
        hla_cut_module = HardCutModule()

    sa_cut_module.to(device)
    hla_cut_module.to(device)


    if _path and _path.is_dir():
        model_dict = torch.load(_path.joinpath('model.pth'), map_location=device)
        epx = model_dict.get('metadata', {}).get('epx', 0)
        print(f"Loading model from {_path}")
        print(model.load_state_dict(model_dict, strict=False))

    else:
        print(f"Generating fresh '{type(model).__name__}' model and align modules.")
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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=20, threshold=0.01, threshold_mode='rel')


    if _path and _path.is_dir() and not load_model_only:
        print(f"Loading optimizer, scheduler, scaler from {_path}")
        optimizer.load_state_dict(torch.load(_path.joinpath('optimizer.pth'), map_location=device))
        scheduler.load_state_dict(torch.load(_path.joinpath('scheduler.pth'), map_location=device))
        scaler.load_state_dict(torch.load(_path.joinpath('scaler.pth'), map_location=device))

    else:
        print(f"Generated fresh optimizer, scheduler, scaler.")

    # for submodule in model.modules():
    #     submodule.register_forward_hook(anomaly_hook)
    # for submodule in sa_atm.modules():
    #     submodule.register_forward_hook(anomaly_hook)
    # for submodule in hla_atm.modules():
    #     submodule.register_forward_hook(anomaly_hook)

    return (model, optimizer, scheduler, scaler), epx



def get_atm(config, num_classes, view, this_script_dir, _path=None):

    assert view in ['sa', 'hla']
    device = config.device

    if view == 'sa':
        affine_path = Path(
        this_script_dir,
        "slice_inflate/preprocessing",
        "mmwhs_1002_SA_yellow_slice_to_ras.mat"
    )
    elif view == 'hla':
        affine_path = Path(
        this_script_dir,
        "slice_inflate/preprocessing",
        "mmwhs_1002_HLA_red_slice_to_ras.mat"
    )
    # Add atm models
    atm = AffineTransformModule(num_classes,
        torch.tensor(config['fov_mm']),
        torch.tensor(config['fov_vox']),
        view_affine=torch.as_tensor(np.loadtxt(affine_path)).float(),
        optim_method=config.affine_theta_optim_method)

    if _path and _path.is_dir():

        atm_dict = torch.load(_path.joinpath(f'{view}_atm.pth'), map_location=device)
        print(f"Loading {view} atm from {_path}")
        print(atm.load_state_dict(sa_atm_dict, strict=False))

    return atm

class NoneOptimizer():
    def __init__(self):
        super().__init__()
    def step():
        pass
    def zero_grad():
        pass

def get_transform_model(config, num_classes, this_script_dir, _path=None, sa_atm_override=None, hla_atm_override=None):
    device = config.device

    if isinstance(sa_atm_override, AffineTransformModule):
        # Check if atm is set externally
        sa_atm = sa_atm_override
    else:
        sa_atm = get_atm(config, num_classes, view='sa', this_script_dir=this_script_dir, _path = _path)

    if isinstance(hla_atm_override, AffineTransformModule):
        # Check if atm is set externally
        hla_atm = hla_atm_override
    else:
        hla_atm = get_atm(config, num_classes, view='hla', this_script_dir=this_script_dir, _path = _path)

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
        transform_parameters = list(sa_atm.parameters())
    elif config.cuts_mode == 'sa>hla':
        transform_parameters = list(hla_atm.parameters())
    elif config.cuts_mode == 'sa+hla':
       transform_parameters = list(sa_atm.parameters()) + list(hla_atm.parameters())
    else:
        raise ValueError()

    if config.train_affine_theta:
        transform_optimizer = torch.optim.AdamW(transform_parameters, weight_decay=0.1, lr=0.001)
    else:
        transform_optimizer = NoneOptimizer()

    if _path and _path.is_dir() and not load_model_only:
        print(f"Loading transform optimizer from {_path}")
        transform_optimizer.load_state_dict(torch.load(_path.joinpath('transform_optimizer.pth'), map_location=device))

    else:
        print(f"Generated fresh transform optimizer.")

    return (sa_atm, hla_atm, sa_cut_module, hla_cut_module), transform_optimizer



# %%
def get_transformed(label, nifti_affine, augment_affine, atm, cut_module,
    crop_around_3d_label_center, crop_around_2d_label_center, image=None):

    img_is_invalid = image is None or image.dim() == 0

    B, num_classes, D, H, W = label.shape

    if img_is_invalid:
        image = torch.zeros(B,1,D,H,W, device=label.device)

    # Transform  label with 'bilinear' interpolation to have gradients
    label = label.float() # TODO Check, can this be removed?
    label.requires_grad = True # TODO Check, can this be removed?
    soft_label, _, _ = atm(label.view(B, num_classes, D, H, W), label.view(B, num_classes, D, H, W),
                            nifti_affine, augment_affine)

    image, label, affine = atm(image.view(B, 1, D, H, W), label.view(B, num_classes, D, H, W),
                                nifti_affine, augment_affine, theta_override=atm.last_theta)

    if crop_around_3d_label_center is not None:
        _3d_vox_size = torch.as_tensor(
            crop_around_3d_label_center)
        label, image, _ = crop_around_label_center(
            label, _3d_vox_size, image)
        _, soft_label, _ = crop_around_label_center(
            label, _3d_vox_size, soft_label)

    label_slc = cut_module(soft_label)
    image_slc = HardCutModule()(image)

    if crop_around_2d_label_center is not None:
        _2d_vox_size = torch.as_tensor(
            crop_around_2d_label_center+[1])
        label_slc, image_slc, _ = crop_around_label_center(
            label_slc, _2d_vox_size, image_slc)

    if img_is_invalid:
        image = torch.empty([])
        image_slc = torch.empty([])
        # Do not set label_slc to .int() here, since we (may) need the gradients
    return image, label.int(), image_slc, label_slc, affine.float()



def get_model_input(batch, config, num_classes, sa_atm, hla_atm, sa_cut_module, hla_cut_module):

    W_TARGET_LEN = 128 # TODO remove this parameter

    b_label = batch['label']
    b_image = batch['image']
    nifti_affine = batch['additional_data']['nifti_affine']
    augment_affine = batch['additional_data']['augment_affine']

    b_label = eo.rearrange(F.one_hot(b_label, num_classes),
                        'B D H W OH -> B OH D H W')
    B,NUM_CLASSES,D,H,W = b_label.shape

    sa_atm.with_batch_theta = config.train_affine_theta
    hla_atm.with_batch_theta = config.train_affine_theta

    sa_image, sa_label, sa_image_slc, sa_label_slc, sa_affine = \
        get_transformed(
            b_label.view(B, NUM_CLASSES, D, H, W),
            nifti_affine, augment_affine,
            sa_atm, sa_cut_module,
            config['crop_around_3d_label_center'], config['crop_around_2d_label_center'],
            image=None)

    hla_image, hla_label, hla_image_slc, hla_label_slc, hla_affine = \
        get_transformed(
            b_label.view(B, NUM_CLASSES, D, H, W),
            nifti_affine, augment_affine,
            hla_atm, hla_cut_module,
            config['crop_around_3d_label_center'], config['crop_around_2d_label_center'],
            image=None)

    if config.cuts_mode == 'sa':
        slices = [sa_label_slc, sa_label_slc]
    elif config.cuts_mode == 'sa>hla':
        slices = [sa_label_slc.detach(), hla_label_slc]
    elif config.cuts_mode == 'sa+hla':
        slices = [hla_label_slc, sa_label_slc]
    else:
        raise ValueError()

    if 'hybrid' in config.model_type:
        b_input = torch.cat(slices, dim=1)
        b_input = b_input.view(-1, NUM_CLASSES*2, W_TARGET_LEN, W_TARGET_LEN)
    else:
        b_input = torch.cat(slices, dim=-1)
        b_input = torch.cat([b_input] * int(W_TARGET_LEN/b_input.shape[-1]), dim=-1) # Stack data hla/sa next to each other

    b_input = b_input.to(device=config.device)
    b_label = b_label.to(device=config.device)

    return b_input.float(), b_label, sa_affine




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

def model_step(config, epx, model, sa_atm, hla_atm, sa_cut_module, hla_cut_module, batch, label_tags, class_weights, io_normalisation_values, autocast_enabled=False):
    # b_input = b_input-io_normalisation_values['input_mean'].to(b_input.device)
    # b_input = b_input/io_normalisation_values['input_std'].to(b_input.device)

    ### Forward pass ###
    with amp.autocast(enabled=autocast_enabled):
        b_input, b_target, _ = get_model_input(batch, config, len(label_tags), sa_atm, hla_atm, sa_cut_module, hla_cut_module)

        wanted_input_dim = 4 if 'hybrid' in config.model_type else 5
        assert b_input.dim() == wanted_input_dim, \
            f"Input image for model must be {wanted_input_dim}D but is {b_input.shape}"

        if config.model_type == 'vae':
            y_hat, (z, mean, std) = model(b_input)
        elif config.model_type in ['ae', 'unet', 'unet-wo-skip', 'hybrid-ae', 'hybrid-unet-wo-skip']:
            y_hat, _ = model(b_input)
        else:
            raise ValueError
        # Reverse normalisation to outputs
        # y_hat = y_hat*io_normalisation_values['target_std'].to(b_input.device)
        # y_hat = y_hat+io_normalisation_values['target_mean'].to(b_input.device)

        ### Calculate loss ###
        assert y_hat.dim() == 5, \
            f"Input shape for loss must be {5}D: BxNUM_CLASSESxSPATIAL but is {y_hat.shape}"
        assert b_target.dim() == 5, \
            f"Target shape for loss must be {5}D: BxNUM_CLASSESxSPATIAL but is {b_target.shape}"

        if "vae" in type(model).__name__.lower():
            loss = get_vae_loss_value(y_hat, b_target.float(), z, mean, std, class_weights, model)
        else:
            loss = get_ae_loss_value(y_hat, b_target.float(), class_weights)

        if config.do_output and epx % 10 == 0 and '1010-mr' in batch['id']:
            idx = batch['id'].index('1010-mr')
            _dir = Path(f"data/output/{NOW_STR}_{wandb.run.name}")
            _dir.mkdir(exist_ok=True)
            nib.save(nib.Nifti1Image(b_input[idx].argmax(0).int().detach().cpu().numpy(), affine=np.eye(4)), _dir.joinpath(f"input_epx_{epx}.nii.gz"))

    return y_hat, b_target, loss



def epoch_iter(epx, global_idx, config, model, sa_atm, hla_atm, sa_cut_module, hla_cut_module, dataset, dataloader, class_weights, fold_postfix, phase='train',
    autocast_enabled=False, all_optimizers=None, scaler=None, store_net_output_to=None):
    PHASES = ['train', 'val', 'test']
    assert phase in ['train', 'val', 'test'], f"phase must be one of {PHASES}"

    epx_losses = []
    epx_sa_thetas = []
    epx_hla_thetas = []
    label_scores_epoch = {}
    seg_metrics_nanmean = {}
    seg_metrics_std = {}
    seg_metrics_nanmean_oa = {}
    seg_metrics_std_oa = {}

    if phase == 'train':
        model.train()
        sa_atm.train()
        hla_atm.train()
        dataset.train(augment=True)
    else:
        model.eval()
        sa_atm.eval()
        hla_atm.eval()
        dataset.eval()

    if isinstance(model, BlendowskiVAE):
        model.set_epoch(epx)

    for batch_idx, batch in tqdm(enumerate(dataloader), desc=phase, total=len(dataloader)):
        if phase == 'train':
            for opt in all_optimizers.values():
                opt.zero_grad()

            y_hat, b_target, loss = model_step(
                config, epx,
                model, sa_atm, hla_atm, sa_cut_module, hla_cut_module,
                batch,
                dataset.label_tags, class_weights,
                dataset.io_normalisation_values, autocast_enabled)

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
                y_hat, b_target, loss = model_step(
                    config, epx,
                    model, sa_atm, hla_atm, sa_cut_module, hla_cut_module,
                    batch,
                    dataset.label_tags, class_weights, dataset.io_normalisation_values, autocast_enabled)

        epx_losses.append(loss.item())
        epx_sa_thetas.append(sa_atm.last_theta_a)
        epx_hla_thetas.append(hla_atm.last_theta_a)

        pred_seg = y_hat.argmax(1)

        # Taken from nibabel nifti1.py
        RZS = sa_atm.last_resampled_affine[0,:3,:3].detach().cpu().numpy()
        nifti_zooms = np.sqrt(np.sum(RZS * RZS, axis=0))

        # Calculate fast dice score
        b_dice = dice3d(
            eo.rearrange(torch.nn.functional.one_hot(pred_seg, len(training_dataset.label_tags)), 'b d h w oh -> b oh d h w'),
            b_target,
            one_hot_torch_style=False
        )
        label_scores_epoch = get_batch_score_per_label(label_scores_epoch, 'dice',
            b_dice, training_dataset.label_tags, exclude_bg=True)

        if epx % 20 == 0 and epx > 0 and False:
            b_hd = hausdorff3d(b_input, b_target, spacing_mm=tuple(nifti_zooms), percent=100)
            label_scores_epoch = get_batch_score_per_label(label_scores_epoch, 'hd',
                b_hd, training_dataset.label_tags, exclude_bg=True)

            b_hd95 = hausdorff3d(b_input, b_target, spacing_mm=tuple(nifti_zooms), percent=95)
            label_scores_epoch = get_batch_score_per_label(label_scores_epoch, 'hd95',
                b_hd95, training_dataset.label_tags, exclude_bg=True)

        if store_net_output_to not in ["", None]:
            store_path = Path(store_net_output_to, f"output_batch{batch_idx}.pth")
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
    log_id = f'losses/{phase}_loss{fold_postfix}'
    log_val = loss_mean
    wandb.log({log_id: log_val}, step=global_idx)
    print(f'losses/{phase}_loss{fold_postfix}', log_val)

    log_label_metrics(f"scores/{phase}_mean", fold_postfix, seg_metrics_nanmean, global_idx,
        logger_selected_metrics=('dice', 'hd', 'hd95'), print_selected_metrics=('dice'))

    log_label_metrics(f"scores/{phase}_std", fold_postfix, seg_metrics_std, global_idx,
        logger_selected_metrics=('dice', 'hd', 'hd95'), print_selected_metrics=())

    log_oa_metrics(f"scores/{phase}_mean_oa_exclude_bg", fold_postfix, seg_metrics_nanmean_oa, global_idx,
        logger_selected_metrics=('dice', 'hd', 'hd95'), print_selected_metrics=('dice', 'hd', 'hd95'))

    log_oa_metrics(f"scores/{phase}_std_oa_exclude_bg", fold_postfix, seg_metrics_std_oa, global_idx,
        logger_selected_metrics=('dice', 'hd', 'hd95'), print_selected_metrics=())

    print()

    mean_transform_dict = dict()

    if epx_sa_thetas:
        print("theta SA rotation param stats are:")
        epx_sa_thetas = torch.cat(epx_sa_thetas).cpu().detach()[:, :3]
        sa_angles = get_theta_params(epx_sa_thetas)[0]
        sa_angles_mean = sa_angles.mean(0)
        sa_angles_std = sa_angles.std(0)

        sa_offsets = get_theta_params(epx_sa_thetas)[1]
        sa_offsets_mean = sa_offsets.mean(0)
        sa_offsets_std = sa_offsets.std(0)
        print("Angles", "mean=", sa_angles_mean, "std=", sa_angles_std)
        print("Offsets", "mean=", sa_offsets_mean, "std=", sa_offsets_std)

        wandb.log({f"orientations/{phase}_sa_angle_mean[1]": sa_angles_mean[1]}, step=global_idx)
        wandb.log({f"orientations/{phase}_sa_angle_std[1]": sa_angles_std[1]}, step=global_idx)
        wandb.log({f"orientations/{phase}_sa_angle_mean[2]": sa_angles_mean[2]}, step=global_idx)
        wandb.log({f"orientations/{phase}_sa_angle_std[2]": sa_angles_std[2]}, step=global_idx)

        wandb.log({f"orientations/{phase}_sa_offset_mean[0]": sa_angles_mean[0]}, step=global_idx)
        wandb.log({f"orientations/{phase}_sa_offset_std[0]": sa_angles_std[0]}, step=global_idx)
        print()

        mean_transform_dict.update(
            dict(
                epoch_sa_angles_mean=sa_angles_mean,
                epoch_sa_offsets_mean=sa_offsets_mean,
            )
        )

    if epx_hla_thetas:
        print("theta HLA rotation param stats are:")
        epx_hla_thetas = torch.cat(epx_hla_thetas).cpu().detach()[:, :3]
        hla_angles = get_theta_params(epx_hla_thetas)[0]
        hla_angles_mean = hla_angles.mean(0)
        hla_angles_std = hla_angles.std(0)

        hla_offsets = get_theta_params(epx_hla_thetas)[1]
        hla_offsets_mean = hla_offsets.mean(0)
        hla_offsets_std = hla_offsets.std(0)
        print("Angles", "mean=", hla_angles_mean, "std=", hla_angles_std)
        print("Offsets", "mean=", hla_angles_mean, "std=", hla_angles_std)

        wandb.log({f"orientations/{phase}_hla_angle_mean[1]": hla_angles_mean[1]}, step=global_idx)
        wandb.log({f"orientations/{phase}_hla_angle_std[1]": hla_angles_std[1]}, step=global_idx)
        wandb.log({f"orientations/{phase}_hla_angle_mean[2]": hla_angles_mean[2]}, step=global_idx)
        wandb.log({f"orientations/{phase}_hla_angle_std[2]": hla_angles_std[2]}, step=global_idx)

        wandb.log({f"orientations/{phase}_hla_offset_mean[0]": hla_offsets_mean[0]}, step=global_idx)
        wandb.log({f"orientations/{phase}_hla_offset_std[0]": hla_offsets_std[0]}, step=global_idx)
        print()

        mean_transform_dict.update(
            dict(
                epoch_hla_angles_mean=hla_angles_mean,
                epoch_hla_offsets_mean=hla_offsets_mean
            )
        )

    print()
    print()



    return loss_mean, mean_transform_dict



def run_dl(run_name, config, training_dataset, test_dataset, stage=None):
    reset_determinism()

    # Configure folds
    if config.num_folds < 1:
        train_idxs = range(training_dataset.__len__(use_2d_override=False))
        val_idxs = []
        fold_idx = -1
        fold_iter = ([fold_idx, (train_idxs, val_idxs)],)

    else:
        kf = KFold(n_splits=config.num_folds)
        fold_iter = enumerate(kf.split(range(training_dataset.__len__(use_2d_override=False))))

        if config.get('fold_override', None):
            selected_fold = config.get('fold_override', 0)
            fold_iter = list(fold_iter)[selected_fold:selected_fold+1]

    fold_means_no_bg = []

    for fold_idx, (train_idxs, val_idxs) in fold_iter:
        fold_postfix = f'_fold{fold_idx}' if fold_idx != -1 else ""

        best_quality_metric = 1.e16
        train_idxs = torch.tensor(train_idxs)
        val_idxs = torch.tensor(val_idxs)
        val_ids = training_dataset.switch_3d_identifiers(val_idxs)

        print(f"Will run validation with these 3D samples (#{len(val_ids)}):", sorted(val_ids))

        ### Add train sampler and dataloaders ##
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idxs)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idxs)
        test_subsampler = torch.utils.data.SubsetRandomSampler(range(len(test_dataset)))

        if not run_test_once_only:
            train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size,
                sampler=train_subsampler, pin_memory=False, drop_last=False,
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
        chk_path = config.checkpoint_path if 'checkpoint_path' in config else None

        (model, optimizer, scheduler, scaler), epx_start = get_model(
            config, len(training_dataset), len(training_dataset.label_tags),
            THIS_SCRIPT_DIR=THIS_SCRIPT_DIR, _path=chk_path, load_model_only=False,
            encoder_training_only=config.encoder_training_only)

        (sa_atm, hla_atm, sa_cut_module, hla_cut_module), transform_optimizer = get_transform_model(
            config, len(training_dataset.label_tags), THIS_SCRIPT_DIR, _path=chk_path,
            sa_atm_override=stage['sa_atm'], hla_atm_override=stage['hla_atm'])

        all_optimizers = dict(optimizer=optimizer, transform_optimizer=transform_optimizer)

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
            print(f"### Log epoch {epx}")
            wandb.log({"ref_epoch_idx": epx}, step=global_idx)

            if not run_test_once_only:
                train_loss, mean_transform_dict = epoch_iter(epx, global_idx, config, model, sa_atm, hla_atm, sa_cut_module, hla_cut_module, training_dataset, train_dataloader, class_weights, fold_postfix,
                    phase='train', autocast_enabled=autocast_enabled, all_optimizers=all_optimizers, scaler=scaler, store_net_output_to=None)

                if stage:
                    stage.update(mean_transform_dict)

                val_loss, _ = epoch_iter(epx, global_idx, config, model, sa_atm, hla_atm, sa_cut_module, hla_cut_module, training_dataset, val_dataloader, class_weights, fold_postfix,
                    phase='val', autocast_enabled=autocast_enabled, all_optimizers=None, scaler=None, store_net_output_to=None)

            quality_metric, _ = test_loss, _ = epoch_iter(epx, global_idx, config, model, sa_atm, hla_atm, sa_cut_module, hla_cut_module, test_dataset, test_dataloader, class_weights, fold_postfix,
                phase='test', autocast_enabled=autocast_enabled, all_optimizers=None, scaler=None, store_net_output_to=config.test_only_and_output_to)

            if run_test_once_only:
                break

            ###  Scheduler management ###
            if config.use_scheduling:
                scheduler.step(quality_metric)

            wandb.log({f'training/scheduler_lr': scheduler.optimizer.param_groups[0]['lr']}, step=global_idx)
            print()

            # Save model
            if config.save_every is None:
                pass

            elif config.save_every == 'best':
                if quality_metric < best_quality_metric:
                    best_quality_metric = quality_metric
                    save_path = f"{config.mdl_save_prefix}/{NOW_STR}_{wandb.run.name}_{fold_postfix}_best"
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
                save_path = f"{config.mdl_save_prefix}/{NOW_STR}_{wandb.run.name}_{fold_postfix}_epx{epx}"
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

        # End of fold loop
        if config.debug or run_test_once_only:
            break

# %%
# training_dataset.eval()
# eval_dataloader = DataLoader(training_dataset, batch_size=20,  pin_memory=False, drop_last=False)

# for large_batch in eval_dataloader:
#     large_b_input = get_model_input(large_batch, config_dict, num_classes=len(training_dataset.label_tags))

# input_mean, input_std = large_b_input[0].float().mean((0,-3,-2,-1), keepdim=True).cpu(), large_b_input[0].float().std((0,-3,-2,-1), keepdim=True).cpu()
# target_mean, target_std = large_b_input[1].float().mean((0,-3,-2,-1), keepdim=True).cpu(), large_b_input[1].float().std((0,-3,-2,-1), keepdim=True).cpu()

# print(input_mean.shape, input_std.shape)
# print(target_mean.shape, target_std.shape)

# torch.save(dict(input_mean=input_mean, input_std=input_std, target_mean=target_mean, target_std=target_std), "io_normalisation_values.pth")
# sys.exit(0)
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
def normal_run():
    with wandb.init(project=PROJECT_NAME, group="training", job_type="train",
            config=config_dict, settings=wandb.Settings(start_method="thread"),
            mode=config_dict['wandb_mode']
        ) as run:

        run_name = run.name
        print("Running", run_name)
        config = wandb.config

        run_dl(run_name, config, training_dataset, test_dataset)



def stage_sweep_run(all_config_dicts, all_stages):
    stage_run_prefix = None

    for config_dict, stage in zip(all_config_dicts, all_stages):
        stg_idx = all_stages.idx

        # Prepare stage settings
        stage.activate()
        print()

        with wandb.init(project=PROJECT_NAME, config=config_dict, settings=wandb.Settings(start_method="thread"),
            mode=config_dict['wandb_mode']) as run:

            if stage_run_prefix is None:
                stage_run_prefix = run.name

            run.name = f"{stage_run_prefix}-stage-{stg_idx+1}"
            print("Running", run.name)
            config = wandb.config

            run_dl(run.name, config, training_dataset, test_dataset, stage)



def wandb_sweep_run():
    with wandb.init(
            settings=wandb.Settings(start_method="thread"),
            mode=config_dict['wandb_mode']) as run:

        run_name = run.name
        print("Running", run_name)
        config = wandb.config

        run_dl(run_name, config, training_dataset, test_dataset)



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



if config_dict['sweep_type'] is None:
    normal_run()

elif config_dict['sweep_type'] == 'wandb_sweep':
    merged_sweep_config_dict = clean_sweep_dict(config_dict)
    sweep_id = wandb.sweep(merged_sweep_config_dict, project=PROJECT_NAME)
    wandb.agent(sweep_id, function=wandb_sweep_run)

elif config_dict['sweep_type'] == 'stage_sweep':
    r_params = init_regularization_params(
        [
            'hla_angles',
            'hla_offsets',
            'sa_angles',
            'sa_offsets',
        ], lambda_r=0.01)

    std_stages = [
        Stage(
            sa_atm=get_atm(config_dict, len(training_dataset.label_tags), 'sa', THIS_SCRIPT_DIR),
            hla_atm=get_atm(config_dict, len(training_dataset.label_tags), 'hla', THIS_SCRIPT_DIR),
            r_params=r_params,
            cuts_mode='sa',
            epochs=35,
            do_output=False,
            __activate_fn__=optimize_sa_angles
        ),
        Stage(
            sa_atm=get_atm(config_dict, len(training_dataset.label_tags), 'sa', THIS_SCRIPT_DIR),
            do_output=False,
            __activate_fn__=optimize_sa_angles
        ),
        Stage(
            sa_atm=get_atm(config_dict, len(training_dataset.label_tags), 'sa', THIS_SCRIPT_DIR),
            do_output=False,
            __activate_fn__=optimize_sa_offsets
        ),
        Stage(
            sa_atm=get_atm(config_dict, len(training_dataset.label_tags), 'sa', THIS_SCRIPT_DIR),
            do_output=False,
            __activate_fn__=optimize_sa_offsets
        ),
        Stage(
            hla_atm=get_atm(config_dict, len(training_dataset.label_tags), 'hla', THIS_SCRIPT_DIR),
            cuts_mode='sa>hla',
            do_output=False,
            __activate_fn__=optimize_hla_angles
        ),
        Stage(
            hla_atm=get_atm(config_dict, len(training_dataset.label_tags), 'hla', THIS_SCRIPT_DIR),
            do_output=False,
            __activate_fn__=optimize_hla_angles
        ),
        Stage(
            hla_atm=get_atm(config_dict, len(training_dataset.label_tags), 'hla', THIS_SCRIPT_DIR),
            do_output=False,
            __activate_fn__=optimize_hla_angles
        ),
        Stage(
            hla_atm=get_atm(config_dict, len(training_dataset.label_tags), 'hla', THIS_SCRIPT_DIR),
            do_output=True,
            __activate_fn__=optimize_hla_offsets
        ),
    ]

    # sa_angle_only_stages = [
    #     Stage(
    #         sa_atm=AffineTransformModule(),
    #         hla_atm=AffineTransformModule(),
    #         r_params=r_params,
    #         cuts_mode='sa',
    #         epx=EPX//4,
    #         do_output=True,
    #         __activate_fn__=optimize_sa_angles
    #     ),
    #     Stage(
    #         sa_atm=get_atm(config_dict, len(training_dataset.label_tags), 'sa', THIS_SCRIPT_DIR),
    #         do_output=True,
    #         __activate_fn__=optimize_sa_angles
    #     ),
    #     Stage(
    #         sa_atm=get_atm(config_dict, len(training_dataset.label_tags), 'sa', THIS_SCRIPT_DIR),
    #         do_output=True,
    #         __activate_fn__=optimize_sa_angles
    #     ),
    #     Stage(
    #         sa_atm=get_atm(config_dict, len(training_dataset.label_tags), 'sa', THIS_SCRIPT_DIR),
    #         do_output=True,
    #         __activate_fn__=optimize_sa_angles
    #     ),
    # ]

    selected_stages = std_stages

    all_config_dicts = []
    for stg in selected_stages:
        # Prepare config dict for the stage
        stage_config = config_dict.copy()
        # Update intersecting keys of both
        stage_config.update((key, stg[key]) for key in set(stg).intersection(stage_config))
        all_config_dicts.append(stage_config)

    stage_sweep_run(all_config_dicts, StageIterator(selected_stages, verbose=True))

else:
    raise ValueError()

# %%
if not in_notebook():
    sys.exit(0)

# %%
# Do any postprocessing / visualization in notebook here

# %%
