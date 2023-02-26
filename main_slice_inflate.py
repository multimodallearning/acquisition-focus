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
from git import Repo
import joblib

from slice_inflate.utils.common_utils import get_script_dir
THIS_SCRIPT_DIR = get_script_dir()

os.environ['MMWHS_CACHE_PATH'] = str(Path(THIS_SCRIPT_DIR, '.cache'))

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
from slice_inflate.utils.log_utils import get_global_idx, log_label_metrics, \
    log_oa_metrics, log_affine_param_stats, log_frameless_image
from sklearn.model_selection import KFold
import numpy as np
import monai

from slice_inflate.losses.dice_loss import DC_and_CE_loss
from slice_inflate.datasets.mmwhs_dataset import MMWHSDataset, load_data, extract_2d_data
from slice_inflate.utils.common_utils import DotDict, in_notebook
from slice_inflate.utils.torch_utils import reset_determinism, ensure_dense, \
    get_batch_dice_over_all, get_batch_score_per_label, save_model, \
    reduce_label_scores_epoch, get_test_func_all_parameters_updated, anomaly_hook
from slice_inflate.models.nnunet_models import Generic_UNet_Hybrid
from slice_inflate.models.affine_transform import AffineTransformModule, get_random_angles, SoftCutModule, HardCutModule, get_theta_params
from slice_inflate.models.ae_models import BlendowskiAE, BlendowskiVAE, HybridAE
from slice_inflate.losses.regularization import optimize_sa_angles, optimize_sa_offsets, optimize_hla_angles, optimize_hla_offsets, init_regularization_params, deactivate_r_params, Stage, StageIterator

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
    args = [config.data_base_path]
    kwargs = dict(
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
        use_binarized_labels=config.use_binarized_labels,
        crop_around_2d_label_center=config.crop_around_2d_label_center,
        max_load_3d_num=config.max_load_3d_num,
        soft_cut_std=config.soft_cut_std,
        augment_angle_std=5,
        device=config.device,
        debug=config.debug
    )

    arghash = joblib.hash(joblib.hash(args)+joblib.hash(kwargs))
    cache_path = Path(os.environ['MMWHS_CACHE_PATH'], arghash, 'dataset.dil')
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.is_file():
        print("Loading dataset from cache:", cache_path)
        with open(cache_path, 'rb') as file:
            dataset = dill.load(file)
    else:
        dataset = MMWHSDataset(*args, **kwargs)
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


    if _path:
        assert Path(_path).is_dir()
        model_dict = torch.load(Path(_path).joinpath('model.pth'), map_location=device)
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
        optim_method=config.affine_theta_optim_method, tag=view)

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
    elif config.cuts_mode == 'hla':
        transform_parameters = list(hla_atm.parameters())
    elif config.cuts_mode == 'sa>hla':
        transform_parameters = list(hla_atm.parameters())
    elif config.cuts_mode == 'sa+hla':
       transform_parameters = list(sa_atm.parameters()) + list(hla_atm.parameters())
    else:
        raise ValueError()

    if config.train_affine_theta:
        assert config.use_affine_theta

    if config.train_affine_theta:
        transform_optimizer = torch.optim.AdamW(transform_parameters, weight_decay=0.1, lr=0.001)
    else:
        transform_optimizer = NoneOptimizer()

    if _path and config.train_affine_theta:
        assert Path(_path).is_dir()
        print(f"Loading transform optimizer from {_path}")
        transform_optimizer.load_state_dict(torch.load(Path(_path).joinpath('transform_optimizer.pth'), map_location=device))

    else:
        print(f"Generated fresh transform optimizer.")

    return (sa_atm, hla_atm, sa_cut_module, hla_cut_module), transform_optimizer



# %%
def get_transformed(label, soft_label, nifti_affine, augment_affine, atm, cut_module,
    crop_around_3d_label_center, crop_around_2d_label_center, image=None):

    img_is_invalid = image is None or image.dim() == 0

    B, num_classes, D, H, W = label.shape

    if img_is_invalid:
        image = torch.zeros(B,1,D,H,W, device=label.device)

    # Transform  label with 'bilinear' interpolation to have gradients
    # label = label.float() # TODO Check, can this be removed?
    # label.requires_grad = True # TODO Check, can this be removed?
    soft_label, _, _ = atm(soft_label.view(B, num_classes, D, H, W), None,
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

    b_label = eo.rearrange(F.one_hot(b_label, num_classes),
                        'B D H W OH -> B OH D H W')
    B,NUM_CLASSES,D,H,W = b_label.shape

    if config.use_distance_map_localization:
        b_soft_label = batch['additional_data']['label_distance_map']
    else:
        b_soft_label = b_label

    nifti_affine = batch['additional_data']['nifti_affine']
    augment_affine = batch['additional_data']['augment_affine']

    sa_atm.use_affine_theta = config.use_affine_theta
    hla_atm.use_affine_theta = config.use_affine_theta

    sa_image, sa_label, sa_image_slc, sa_label_slc, sa_affine = \
        get_transformed(
            b_label.view(B, NUM_CLASSES, D, H, W),
            b_soft_label.view(B, NUM_CLASSES, D, H, W),
            nifti_affine, augment_affine,
            sa_atm, sa_cut_module,
            config['crop_around_3d_label_center'], config['crop_around_2d_label_center'],
            image=None)

    hla_image, hla_label, hla_image_slc, hla_label_slc, hla_affine = \
        get_transformed(
            b_label.view(B, NUM_CLASSES, D, H, W),
            b_soft_label.view(B, NUM_CLASSES, D, H, W),
            nifti_affine, augment_affine,
            hla_atm, hla_cut_module,
            config['crop_around_3d_label_center'], config['crop_around_2d_label_center'],
            image=None)

    if config.cuts_mode == 'sa':
        slices = [sa_label_slc, sa_label_slc]
    elif config.cuts_mode == 'hla':
        slices = [hla_label_slc, hla_label_slc]
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

    if config.reconstruction_target == 'from-dataloader':
        b_target = b_label
    elif config.reconstruction_target == 'sa-oriented':
        b_target = sa_label
    elif config.reconstruction_target == 'hla-oriented':
        b_target = hla_label
    else:
        raise ValueError()

    b_input = b_input.to(device=config.device)
    b_target = b_target.to(device=config.device)

    return b_input.float(), b_target, sa_affine




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
        elif config.model_type in ['ae', 'unet', 'hybrid-unet', 'unet-wo-skip', 'hybrid-ae', 'hybrid-unet-wo-skip']:
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

    return y_hat, b_target, loss, b_input



def epoch_iter(epx, global_idx, config, model, sa_atm, hla_atm, sa_cut_module, hla_cut_module, dataset, dataloader, class_weights, fold_postfix, phase='train',
    autocast_enabled=False, all_optimizers=None, scaler=None, store_net_output_to=None, r_params=None):
    PHASES = ['train', 'val', 'test']
    assert phase in ['train', 'val', 'test'], f"phase must be one of {PHASES}"

    epx_losses = []

    epx_sa_theta_aps = {}
    epx_hla_theta_aps = {}
    epx_sa_theta_tps = {}
    epx_hla_theta_tps = {}
    epx_input = {}

    label_scores_epoch = {}
    seg_metrics_nanmean = {}
    seg_metrics_std = {}
    seg_metrics_nanmean_oa = {}
    seg_metrics_std_oa = {}

    if phase == 'train':
        model.train()
        sa_atm.train()
        hla_atm.train()
        dataset.train(augment=config.do_augment)
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

            y_hat, b_target, loss, b_input = model_step(
                config, epx,
                model, sa_atm, hla_atm, sa_cut_module, hla_cut_module,
                batch,
                dataset.label_tags, class_weights,
                dataset.io_normalisation_values, autocast_enabled)

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
                    dataset.label_tags, class_weights, dataset.io_normalisation_values, autocast_enabled)

        epx_losses.append(loss.item())

        epx_input.update({k:v for k,v in zip(batch['id'], b_input)})

        if sa_atm.last_theta_ap is not None:
            epx_sa_theta_aps.update({k:v for k,v in zip(batch['id'], sa_atm.last_theta_ap)})
        if sa_atm.last_theta_tp is not None:
            epx_sa_theta_tps.update({k:v for k,v in zip(batch['id'], sa_atm.last_theta_tp)})
        if hla_atm.last_theta_ap is not None:
            epx_hla_theta_aps.update({k:v for k,v in zip(batch['id'], hla_atm.last_theta_ap)})
        if hla_atm.last_theta_tp is not None:
            epx_hla_theta_tps.update({k:v for k,v in zip(batch['id'], hla_atm.last_theta_tp)})

        pred_seg = y_hat.argmax(1)

        # Taken from nibabel nifti1.py
        rzs = sa_atm.last_resampled_affine[0,:3,:3]
        nifti_zooms = (rzs[:3,:3]*rzs[:3,:3]).sum(1).sqrt().detach().cpu()

        # Calculate fast dice score
        pred_seg_oh = eo.rearrange(torch.nn.functional.one_hot(pred_seg, len(training_dataset.label_tags)), 'b d h w oh -> b oh d h w')

        b_dice = monai.metrics.compute_dice(pred_seg_oh, b_target)

        label_scores_epoch = get_batch_score_per_label(label_scores_epoch, 'dice',
            b_dice, training_dataset.label_tags, exclude_bg=True)

        if (epx % 20 == 0 and epx > 0) or (epx+1 == config.epochs) or config.debug:
            b_sz = pred_seg_oh.shape[0]

            b_iou = monai.metrics.compute_iou(pred_seg_oh, b_target)
            label_scores_epoch = get_batch_score_per_label(label_scores_epoch, 'iou',
                b_iou, training_dataset.label_tags, exclude_bg=True)

            b_hd = monai.metrics.compute_hausdorff_distance(pred_seg_oh, b_target) * nifti_zooms.norm()
            b_hd = torch.cat([torch.zeros(b_sz,1), b_hd], dim=1) # Add zero score for background
            label_scores_epoch = get_batch_score_per_label(label_scores_epoch, 'hd',
                b_hd, training_dataset.label_tags, exclude_bg=True)

            b_hd95 = monai.metrics.compute_hausdorff_distance(pred_seg_oh, b_target, percentile=95) * nifti_zooms.norm()
            b_hd95 = torch.cat([torch.zeros(b_sz,1), b_hd95], dim=1) # Add zero score for background
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
        logger_selected_metrics=('dice', 'iou', 'hd', 'hd95'), print_selected_metrics=('dice'))

    log_label_metrics(f"scores/{phase}_std", fold_postfix, seg_metrics_std, global_idx,
        logger_selected_metrics=('dice', 'iou', 'hd', 'hd95'), print_selected_metrics=())

    log_oa_metrics(f"scores/{phase}_mean_oa_exclude_bg", fold_postfix, seg_metrics_nanmean_oa, global_idx,
        logger_selected_metrics=('dice', 'iou', 'hd', 'hd95'), print_selected_metrics=('dice', 'iou', 'hd', 'hd95'))

    log_oa_metrics(f"scores/{phase}_std_oa_exclude_bg", fold_postfix, seg_metrics_std_oa, global_idx,
        logger_selected_metrics=('dice', 'iou', 'hd', 'hd95'), print_selected_metrics=())

    print()

    mean_transform_dict = dict()

    if epx_sa_theta_aps:
        ornt_log_prefix = f"orientations/{phase}_sa_"
        sa_param_dict = dict(
            theta_ap=epx_sa_theta_aps.values(),
            theta_tp=epx_sa_theta_tps.values()
        )
        sa_theta_ap_mean, sa_theta_tp_mean = \
            log_affine_param_stats(ornt_log_prefix, fold_postfix, sa_param_dict, global_idx,
                logger_selected_metrics=('mean', 'std'), print_selected_metrics=('mean', 'std'))
        print()

        mean_transform_dict.update(
            dict(
                epoch_sa_theta_ap_mean=sa_theta_ap_mean,
                epoch_sa_theta_tp_mean=sa_theta_tp_mean,
            )
        )

    if epx_hla_theta_aps:
        ornt_log_prefix = f"orientations/{phase}_hla_"
        hla_param_dict = dict(
            theta_ap=epx_hla_theta_aps.values(),
            theta_tp=epx_hla_theta_tps.values()
        )
        hla_theta_ap_mean, hla_theta_tp_mean = \
            log_affine_param_stats(ornt_log_prefix, fold_postfix, hla_param_dict, global_idx,
                logger_selected_metrics=('mean', 'std'), print_selected_metrics=('mean', 'std'))
        print()

        mean_transform_dict.update(
            dict(
                epoch_hla_theta_ap_mean=hla_theta_ap_mean,
                epoch_hla_theta_tp_mean=hla_theta_tp_mean,
            )
        )

    if config.do_output:
        # Store the slice model input
        save_input = torch.stack(list(epx_input.values()))

        _dir = Path(f"data/output/{wandb.run.name}")
        _dir.mkdir(exist_ok=True)

        if config.use_distance_map_localization:
            save_input =  (save_input < 0.5).float()

        if 'hybrid' in config.model_type:
            num_classes = len(training_dataset.label_tags)
            save_input = save_input.chunk(2,dim=1)
            save_input = torch.cat([slc.argmax(1, keepdim=True) for slc in save_input], dim=1)

        else:
            save_input = save_input.argmax(0)

        save_input = save_input.cpu()
        BI, DI, HI, WI = save_input.shape
        img_input = eo.rearrange(save_input, 'BI DI HI WI -> (DI WI) (BI HI)')
        log_frameless_image(img_input.numpy(), _dir / f"input_{phase}_epx_{epx}.png", dpi=150, cmap='gray')

        lean_dct = {k:v for k,v in zip(epx_input.keys(), save_input.short())}
        torch.save(lean_dct, _dir / f"input_{phase}_epx_{epx}.pt")

    print(f"### END {phase.upper()}")
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
        mdl_chk_path = config.checkpoint_path if 'checkpoint_path' in config else None

        (model, optimizer, scheduler, scaler), epx_start = get_model(
            config, len(training_dataset), len(training_dataset.label_tags),
            THIS_SCRIPT_DIR=THIS_SCRIPT_DIR, _path=mdl_chk_path, load_model_only=False,
            encoder_training_only=config.encoder_training_only)

        # Load transformation model from checkpoint, if any
        transform_mdl_chk_path = config.transform_model_checkpoint_path if 'transform_model_checkpoint_path' in config else None
        sa_atm_override = stage['sa_atm'] if stage is not None and 'sa_atm' in stage else None
        hla_atm_override = stage['hla_atm'] if stage is not None and 'hla_atm' in stage else None

        (sa_atm, hla_atm, sa_cut_module, hla_cut_module), transform_optimizer = get_transform_model(
            config, len(training_dataset.label_tags), THIS_SCRIPT_DIR, _path=transform_mdl_chk_path,
            sa_atm_override=sa_atm_override, hla_atm_override=hla_atm_override)

        all_optimizers = dict(optimizer=optimizer, transform_optimizer=transform_optimizer)
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
                train_loss, mean_transform_dict, b_input = epoch_iter(
                    epx, global_idx, config, model, sa_atm, hla_atm, sa_cut_module, hla_cut_module,
                    training_dataset, train_dataloader, class_weights, fold_postfix,
                    phase='train', autocast_enabled=autocast_enabled,
                    all_optimizers=all_optimizers, scaler=scaler, store_net_output_to=None,
                    r_params=r_params)

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
                    save_path = f"{config.mdl_save_prefix}/{wandb.run.name}_{fold_postfix}_best"
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
                save_path = f"{config.mdl_save_prefix}/{wandb.run.name}_{fold_postfix}_epx{epx}"
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
        run.name = f"{NOW_STR}_{run.name}"
        print("Running", run.name)
        config = wandb.config

        run_dl(run.name, config, training_dataset, test_dataset)



def stage_sweep_run(config_dict, all_stages):
    stage_run_prefix = None

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

            if stage_run_prefix is None:
                stage_run_prefix = run.name

            run.name = f"{NOW_STR}_{stage_run_prefix}-stage-{stg_idx+1}"
            print("Running", run.name)
            config = wandb.config

            run_dl(run.name, config, training_dataset, test_dataset, stage)
        wandb.finish()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(device=0)
        print(f"CUDA memory used: {(total-free)/1024**3:.2f}/{total/1024**3:.2f} GB ({(total-free)/total*100:.2}%)")



def wandb_sweep_run():
    with wandb.init(
            settings=wandb.Settings(start_method="thread"),
            mode=config_dict['wandb_mode']) as run:

        run.name = f"{NOW_STR}_{run.name}"
        print("Running", run.name)
        config = wandb.config

        run_dl(run.name, config, training_dataset, test_dataset)



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
            r_params=r_params,
            sa_atm=get_atm(config_dict, len(training_dataset.label_tags), 'sa', THIS_SCRIPT_DIR),
            hla_atm=get_atm(config_dict, len(training_dataset.label_tags), 'hla', THIS_SCRIPT_DIR),
            do_output=True,
            cuts_mode='sa',
            reconstruction_target='sa-oriented',
            epochs=config_dict['epochs'],
            soft_cut_std=-999,
            train_affine_theta=False,
            __activate_fn__=deactivate_r_params
        ),
        Stage(
            sa_atm=get_atm(config_dict, len(training_dataset.label_tags), 'sa', THIS_SCRIPT_DIR),
            cuts_mode='sa',
            reconstruction_target='from-dataloader',
            epochs=40,
            soft_cut_std=0.125,
            train_affine_theta=True,
            do_output=True,
            __activate_fn__=optimize_sa_angles
        ),
        Stage(
            do_output=True,
            cuts_mode='sa',
            reconstruction_target='sa-oriented',
            epochs=config_dict['epochs'],
            soft_cut_std=-999,
            train_affine_theta=False,
            __activate_fn__=deactivate_r_params
        ),
        Stage(
            sa_atm=get_atm(config_dict, len(training_dataset.label_tags), 'sa', THIS_SCRIPT_DIR),
            cuts_mode='sa',
            reconstruction_target='from-dataloader',
            epochs=40,
            soft_cut_std=0.125,
            do_output=True,
            train_affine_theta=True,
            __activate_fn__=optimize_sa_angles
        ),
        Stage(
            do_output=True,
            cuts_mode='sa',
            reconstruction_target='sa-oriented',
            epochs=config_dict['epochs'],
            soft_cut_std=-999,
            train_affine_theta=False,
            __activate_fn__=deactivate_r_params
        ),
        # Stage(
        #     sa_atm=get_atm(config_dict, len(training_dataset.label_tags), 'sa', THIS_SCRIPT_DIR),
        #     do_output=True,
        #     __activate_fn__=optimize_sa_offsets
        # ),
        # Stage(
        #     hla_atm=get_atm(config_dict, len(training_dataset.label_tags), 'hla', THIS_SCRIPT_DIR),
        #     cuts_mode='sa>hla',
        #     do_output=True,
        #     __activate_fn__=optimize_hla_angles
        # ),
        # Stage(
        #     hla_atm=get_atm(config_dict, len(training_dataset.label_tags), 'hla', THIS_SCRIPT_DIR),
        #     do_output=True,
        #     __activate_fn__=optimize_hla_angles
        # ),
        # Stage(
        #     hla_atm=get_atm(config_dict, len(training_dataset.label_tags), 'hla', THIS_SCRIPT_DIR),
        #     do_output=True,
        #     __activate_fn__=optimize_hla_offsets
        # ),
    ]

    sa_angle_only_stages = [
        Stage(
            r_params=r_params,
            sa_atm=get_atm(config_dict, len(training_dataset.label_tags), 'sa', THIS_SCRIPT_DIR),
            hla_atm=get_atm(config_dict, len(training_dataset.label_tags), 'hla', THIS_SCRIPT_DIR),
            cuts_mode='sa',
            reconstruction_target='from-dataloader',
            epochs=40,
            soft_cut_std=-999,
            train_affine_theta=True,
            do_output=True,
            __activate_fn__=optimize_sa_angles
        ),
        Stage(
            do_output=True,
            cuts_mode='sa',
            reconstruction_target='sa-oriented',
            epochs=config_dict['epochs'],
            soft_cut_std=-999,
            train_affine_theta=False,
            use_distance_map_localization=False,
            __activate_fn__=deactivate_r_params
        ),
    ]

    sa_offset_only_stages = [
        Stage(
            r_params=r_params,
            sa_atm=get_atm(config_dict, len(training_dataset.label_tags), 'sa', THIS_SCRIPT_DIR),
            hla_atm=get_atm(config_dict, len(training_dataset.label_tags), 'hla', THIS_SCRIPT_DIR),
            do_output=True,
            cuts_mode='sa',
            reconstruction_target='from-dataloader',
            epochs=40,
            soft_cut_std=-999,
            train_affine_theta=True,
            __activate_fn__=optimize_sa_offsets
        ),
        Stage(
            do_output=True,
            cuts_mode='sa',
            reconstruction_target='sa-oriented',
            epochs=config_dict['epochs'],
            soft_cut_std=-999,
            train_affine_theta=False,
            __activate_fn__=deactivate_r_params
        ),
    ]

    sa_angle_offset_stages = [
        Stage(
            r_params=r_params,
            sa_atm=get_atm(config_dict, len(training_dataset.label_tags), 'sa', THIS_SCRIPT_DIR),
            hla_atm=get_atm(config_dict, len(training_dataset.label_tags), 'hla', THIS_SCRIPT_DIR),
            cuts_mode='sa',
            reconstruction_target='from-dataloader',
            epochs=50,
            soft_cut_std=-999,
            train_affine_theta=True,
            do_output=True,
            __activate_fn__=optimize_sa_angles
        ),
        Stage(
            sa_atm=get_atm(config_dict, len(training_dataset.label_tags), 'sa', THIS_SCRIPT_DIR),
            hla_atm=get_atm(config_dict, len(training_dataset.label_tags), 'hla', THIS_SCRIPT_DIR),
            do_output=True,
            cuts_mode='sa',
            reconstruction_target='from-dataloader',
            epochs=50,
            soft_cut_std=-999,
            train_affine_theta=True,
            __activate_fn__=optimize_sa_offsets
        ),
        Stage(
            do_output=True,
            cuts_mode='sa',
            reconstruction_target='sa-oriented',
            epochs=config_dict['epochs'],
            soft_cut_std=-999,
            train_affine_theta=False,
            __activate_fn__=deactivate_r_params
        ),
    ]

    sa_all_params_stages = [
        Stage(
            r_params=None,
            sa_atm=get_atm(config_dict, len(training_dataset.label_tags), 'sa', THIS_SCRIPT_DIR),
            hla_atm=get_atm(config_dict, len(training_dataset.label_tags), 'hla', THIS_SCRIPT_DIR),
            cuts_mode='sa',
            reconstruction_target='from-dataloader',
            epochs=40,
            soft_cut_std=-999,
            use_distance_map_localization=True,
            train_affine_theta=True,
            do_output=True,
            __activate_fn__=lambda stage: None
        ),
        Stage(
            do_output=True,
            cuts_mode='sa',
            reconstruction_target='sa-oriented',
            epochs=config_dict['epochs'],
            soft_cut_std=-999,
            train_affine_theta=False,
            use_distance_map_localization=False,
            __activate_fn__=lambda stage: None
        ),
    ]

    all_params_stages = [
        Stage( # Optimize SA
            r_params=r_params,
            sa_atm=get_atm(config_dict, len(training_dataset.label_tags), 'sa', THIS_SCRIPT_DIR),
            hla_atm=get_atm(config_dict, len(training_dataset.label_tags), 'hla', THIS_SCRIPT_DIR),
            cuts_mode='sa',
            reconstruction_target='sa-oriented',
            epochs=40,
            soft_cut_std=-999,
            use_distance_map_localization=False,
            use_affine_theta=True,
            train_affine_theta=True,
            do_output=True,
            __activate_fn__=lambda self: None
        ),
        Stage( # Optimize hla
            hla_atm=get_atm(config_dict, len(training_dataset.label_tags), 'hla', THIS_SCRIPT_DIR),
            cuts_mode='sa>hla',
            reconstruction_target='hla-oriented',
            epochs=40,
            soft_cut_std=-999,
            use_distance_map_localization=False,
            use_affine_theta=True,
            train_affine_theta=True,
            do_output=True,
            __activate_fn__=lambda self: None
        ),
        Stage( # Final optimized run
            do_output=True,
            cuts_mode='sa+hla',
            reconstruction_target='hla-oriented',
            epochs=config_dict['epochs'],
            soft_cut_std=-999,
            use_affine_theta=True,
            train_affine_theta=False,
            use_distance_map_localization=False,
            __activate_fn__=lambda self: None
        ),
        Stage( # Reference run
            do_output=True,
            cuts_mode='sa+hla',
            reconstruction_target='hla-oriented',
            epochs=config_dict['epochs'],
            soft_cut_std=-999,
            train_affine_theta=False,
            use_affine_theta=False,
            use_distance_map_localization=False,
            __activate_fn__=lambda self: None
        ),
    ]


    selected_stages = all_params_stages
    stage_sweep_run(config_dict, StageIterator(selected_stages, verbose=True))

else:
    raise ValueError()

# %%
if not in_notebook():
    sys.exit(0)

# %%
# Do any postprocessing / visualization in notebook here

# %%
