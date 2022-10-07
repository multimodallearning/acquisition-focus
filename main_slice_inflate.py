# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: 'Python 3.9.12 (''.venv'': poetry)'
#     language: python
#     name: python3
# ---

# %%
import os
from pathlib import Path

os.environ['MMWHS_CACHE_PATH'] = str(Path('.', '.cache'))

from meidic_vtach_utils.run_on_recommended_cuda import get_cuda_environ_vars as get_vars
os.environ.update(get_vars("*"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

from tqdm import tqdm
import wandb
import nibabel as nib

from slice_inflate.datasets.mmwhs_dataset import MMWHSDataset, load_data, extract_2d_data
from slice_inflate.utils.common_utils import DotDict, get_script_dir, in_notebook
from slice_inflate.utils.torch_utils import reset_determinism, ensure_dense, get_batch_dice_over_all, get_batch_dice_per_class, save_model
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from slice_inflate.datasets.align_mmwhs import cut_slice
from slice_inflate.utils.log_utils import get_global_idx, log_class_dices
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from mdl_seg_class.metrics import dice3d
import numpy as np
THIS_SCRIPT_DIR = get_script_dir()

PROJECT_NAME = "slice_inflate"

training_dataset, test_dataset = None, None
# %%
config_dict = DotDict({
    'num_folds': 0,
    'state': 'train', #
    # 'fold_override': 0,
    # 'checkpoint_epx': 0,
                   # If true use MIND features (https://pubmed.ncbi.nlm.nih.gov/22722056/)
    'epochs': 500,

    'batch_size': 4,
    'val_batch_size': 1,
    'modality': 'all',
    'use_2d_normal_to': None,               # Can be None or 'D', 'H', 'W'. If not None 2D slices will be selected for training

    'dataset': 'mmwhs',                 # The dataset prepared with our preprocessing scripts
    'data_base_path': str(Path(THIS_SCRIPT_DIR, "data/MMWHS")),
    'reg_state': None, # Registered (noisy) labels used in training. See prepare_data() for valid reg_states
    'train_set_max_len': None,              # Length to cut of dataloader sample count
    'crop_around_3d_label_center': None, #(128,128,128),
    'crop_3d_region': None, #((0,128), (0,128), (0,128)), # dimension range in which 3D samples are cropped
    'crop_2d_slices_gt_num_threshold': 0,   # Drop 2D slices if less than threshold pixels are positive
    'crop_around_2d_label_center': None, #(128,128),

    'lr': 1e-3,
    'use_scheduling': True,
    'model_type': 'ae',

    'save_every': 'best',
    'mdl_save_prefix': 'data/models',

    'debug': False,
    'wandb_mode': 'online',                         # e.g. online, disabled. Use weights and biases online logging
    'do_sweep': False,                                # Run multiple trainings with varying config values defined in sweep_config_dict below

    # For a snapshot file: dummy-a2p2z76CxhCtwLJApfe8xD_fold0_epx0
    'checkpoint_name': None,                          # Training snapshot name, e.g. dummy-a2p2z76CxhCtwLJApfe8xD
    'fold_override': None,                            # Training fold, e.g. 0
    'checkpoint_epx': None,                           # Training epx, e.g. 0

    'do_plot': False,                                 # Generate plots (debugging purpose)
    'save_dp_figures': False,                         # Plot data parameter value distribution
    'save_labels': True,                              # Store training labels alongside data parameter values inside the training snapshot

    'device': 'cuda'
})

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
        crop_around_3d_label_center=config.crop_around_3d_label_center,
        pre_interpolation_factor=1., # When getting the data, resize the data by this factor
        ensure_labeled_pairs=True, # Only use fully labelled images (segmentation label available)
        use_2d_normal_to=config.use_2d_normal_to, # Use 2D slices cut normal to D,H,>W< dimensions
        crop_around_2d_label_center=config.crop_around_2d_label_center,

        augment_angle_std=5,

        device=config.device,
        debug=config.debug
    )

    return training_dataset


# %%
if training_dataset is None:
    training_dataset = prepare_data(config_dict)

if test_dataset is None:
    test_config = config_dict.copy()
    test_config['state'] = 'test'
    test_dataset = prepare_data(DotDict(test_config))

# %%
if False:
    training_dataset.train(augment=True)
    training_dataset.self_attributes['augment_angle_std'] = 5
    print("do_augment", training_dataset.do_augment)
    for sample in [training_dataset[idx] for idx in range(20)]:
        fig = plt.figure(figsize=(16., 4.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
            nrows_ncols=(1, 6),  # creates 2x2 grid of axes
            axes_pad=0.0,  # pad between axes in inch.
        )

        show_row = [
            cut_slice(sample['image']),
            cut_slice(sample['label']),

            sample['sa_image_slc'],
            sample['sa_label_slc'],

            sample['hla_image_slc'],
            sample['hla_label_slc'],
        ]

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

# %%
import contextlib

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


# %%
class BlendowskiAE(torch.nn.Module):

    class ConvBlock(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels_list: list, strides_list: list, kernels_list:list=None, paddings_list:list=None):
            super().__init__()

            ops = []
            in_channels = [in_channels] + out_channels_list[:-1]
            if kernels_list is None:
                kernels_list = [3] * len(out_channels_list)
            if paddings_list is None:
                paddings_list = [1] * len(out_channels_list)

            for op_idx in range(len(out_channels_list)):
                ops.append(torch.nn.Conv3d(
                    in_channels[op_idx],
                    out_channels_list[op_idx],
                    kernel_size=kernels_list[op_idx],
                    stride=strides_list[op_idx],
                    padding=paddings_list[op_idx]
                ))
                ops.append(torch.nn.BatchNorm3d(out_channels_list[op_idx]))
                ops.append(torch.nn.LeakyReLU())

            self.block = torch.nn.Sequential(*ops)

        def forward(self, x):
            return self.block(x)



    def __init__(self, in_channels, out_channels, decoder_in_channels=2, debug_mode=False):
        super().__init__()

        self.debug_mode = debug_mode

        self.first_layer_encoder = self.ConvBlock(in_channels, out_channels_list=[8], strides_list=[1])
        self.first_layer_decoder = self.ConvBlock(8, out_channels_list=[8,out_channels], strides_list=[1,1])

        self.second_layer_encoder = self.ConvBlock(8, out_channels_list=[20,20,20], strides_list=[2,1,1])
        self.second_layer_decoder = self.ConvBlock(20, out_channels_list=[8], strides_list=[1])

        self.third_layer_encoder = self.ConvBlock(20, out_channels_list=[40,40,40], strides_list=[2,1,1])
        self.third_layer_decoder = self.ConvBlock(decoder_in_channels, out_channels_list=[20], strides_list=[1])

        # self.fourth_layer_encoder = self.ConvBlock(40, out_channels_list=[60,60,60], strides_list=[2,1,1])
        # self.fourth_layer_decoder = self.ConvBlock(decoder_in_channels, out_channels_list=[40], strides_list=[1])

        self.deepest_layer = torch.nn.Sequential(
            self.ConvBlock(40, out_channels_list=[40,30,20], strides_list=[2,1,1]),
            torch.nn.Conv3d(20, 2, kernel_size=1, stride=1, padding=0)
        )

        self.encoder = torch.nn.Sequential(
            self.first_layer_encoder,
            self.second_layer_encoder,
            self.third_layer_encoder,
            # self.fourth_layer_encoder,
        )

        self.decoder = torch.nn.Sequential(
            # torch.nn.Upsample(scale_factor=2),
            # self.fourth_layer_decoder,
            torch.nn.Upsample(scale_factor=2),
            self.third_layer_decoder,
            torch.nn.Upsample(scale_factor=2),
            self.second_layer_decoder,
            torch.nn.Upsample(scale_factor=2),
            self.first_layer_decoder,
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.deepest_layer(h)
        # h = debug_forward_pass(self.encoder, x, STEP_MODE=False)
        # h = debug_forward_pass(self.deepest_layer, h, STEP_MODE=False)
        return h

    def decode(self, z):
        if self.debug_mode:
            return debug_forward_pass(self.decoder, z, STEP_MODE=False)
        else:
            return self.decoder(z)

    def forward(self, x):
        # x = torch.nn.functional.instance_norm(x)
        z = self.encode(x)
        return self.decode(z), z



class BlendowskiVAE(BlendowskiAE):
    def __init__(self, *args, **kwargs):
        kwargs['decoder_in_channels'] = 1
        super().__init__(*args, **kwargs)

        self.deepest_layer = nn.ModuleList([
            self.ConvBlock(60, out_channels_list=[60,20,20,1], strides_list=[2,1,1,1], kernels_list=[3,3,3,1], paddings_list=[1,1,1,0]),
            self.ConvBlock(60, out_channels_list=[60,20,20,1], strides_list=[2,1,1,1], kernels_list=[3,3,3,1], paddings_list=[1,1,1,0]),
        ])

        self.log_var_scale = nn.Parameter(torch.Tensor([0.0]))

    def sample_z(self, mean, std):
        return torch.normal(mean=mean, std=std)

    def encode(self, x):
        h = self.encoder(x)
        mean = self.deepest_layer[0](h)
        log_var = self.deepest_layer[1](h)
        return mean, log_var

    def forward(self, x):
        mean, log_var = self.encode(x)

        if self.training:
            std = torch.exp(log_var/2) + 1e-6
        else:
            std = 1e-6 * torch.ones_like(log_var)

        z = self.sample_z(mean=mean, std=std)

        return self.decode(z), (z, mean, std)



# %%
# x = torch.zeros(1,8,128,128,128)
# bae = BlendowskiAE(in_channels=8, out_channels=8)

# y, z = bae(x)

# print("BAE")
# print("x", x.shape)
# print("z", z.shape)
# print("y", y.shape)
# print()

# bvae = BlendowskiVAE(in_channels=8, out_channels=8)

# y, z = bvae(x)

# print("BVAE")
# print("x", x.shape)
# print("z", z.shape)
# print("y", y.shape)


# %%
# model = BlendowskiVAE(in_channels=6, out_channels=6)
# model.cuda()
# with torch.no_grad():
#     smp = torch.nn.functional.one_hot(training_dataset[1]['label'], 6).unsqueeze(0).permute([0,4,1,2,3]).float().cuda()
# y, _ = model(smp)

# %%
# def nan_hook(self, inp, output):
#     if not isinstance(output, tuple):
#         outputs = [output]
#     else:
#         outputs = output

#     for i, out in enumerate(outputs):
#         nan_mask = torch.isnan(out)
#         if nan_mask.any():
#             print("In", self.__class__.__name__)
#             raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

def get_model(config, dataset_len, num_classes, THIS_SCRIPT_DIR, _path=None, device='cpu'):
    _path = Path(THIS_SCRIPT_DIR).joinpath(_path).resolve()

    if config.model_type == 'vae':
        model = BlendowskiVAE(in_channels=num_classes, out_channels=num_classes)
    elif config.model_type == 'ae':
            model = BlendowskiAE(in_channels=num_classes, out_channels=num_classes)
    else:
        raise ValueError
    model.to(device)
    print(f"Param count model: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scaler = amp.GradScaler()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, verbose=True)

    if _path and _path.is_dir():
        print(f"Loading model, optimizers and grad scalers from {_path}")
        model.load_state_dict(torch.load(_path.joinpath('model.pth'), map_location=device))
        optimizer.load_state_dict(torch.load(_path.joinpath('optimizer.pth'), map_location=device))
        scheduler.load_state_dict(torch.load(_path.joinpath('scheduler.pth'), map_location=device))
        scaler.load_state_dict(torch.load(_path.joinpath('scaler.pth'), map_location=device))
    else:
        print(f"Generating fresh '{type(model).__name__}' model, optimizer and grad scaler.")

    # for submodule in model.modules():
    #     submodule.register_forward_hook(nan_hook)

    return (model, optimizer, scheduler, scaler)


# %%
def get_model_input(batch, config, num_classes):
    b_hla_slc_seg = batch['hla_label_slc']
    b_sa_slc_seg = batch['sa_label_slc']
    b_input = torch.cat(
        [b_sa_slc_seg.unsqueeze(1).repeat(1,64,1,1),
            b_hla_slc_seg.unsqueeze(1).repeat(1,64,1,1)],
            dim=1
    )
    b_seg = batch['label']

    b_input = b_input.to(device=config.device)
    b_seg = b_seg.to(device=config.device)

    b_input = F.one_hot(b_input, num_classes).permute(0,4,1,2,3)
    b_input = b_input.float()
    b_seg = F.one_hot(b_seg, num_classes).permute(0,4,1,2,3)

    return b_seg, b_seg

def inference_wrap(model, seg):
    with torch.inference_mode():
        b_seg = seg.unsqueeze(0).unsqueeze(0).float()
        b_out = model(b_seg)[0]
        b_out = b_out.argmax(1)
        return b_out



def gaussian_likelihood(y_hat, log_var_scale, y_target):
    B, C, *_ = y_hat.shape
    mean = y_hat
    scale = torch.exp(log_var_scale/2)
    dist = torch.distributions.Normal(mean, scale)

    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(y_target)

    # GLH
    return log_pxz.reshape(B, C, -1).mean(-1)



def kl_divergence(z, mean, std):
    # See https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    B,C, *_ = z.shape
    p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
    q = torch.distributions.Normal(mean, std)

    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # KL divergence
    kl = (log_qzx - log_pz)

    # Reduce spatial dimensions
    kl = kl.view(-1).mean(-1)
    return kl



def get_ae_loss_value(y_hat, y_target, class_weights):
    B, *_ = y_target.shape
    # y_target = (y_target/y_target.std((-1,-2,-3)).view(B,6,1,1,1))
    # y_target = (y_target-y_target.mean((-1,-2,-3)).view(B,6,1,1,1))
    return nn.CrossEntropyLoss(class_weights)(y_hat, y_target)


def get_vae_loss_value(y_hat, y_target, z, mean, std, class_weights, model):
    # Reconstruction loss
    # recon_loss = gaussian_likelihood(y_hat, model.log_var_scale, y_target.float()) # TODO Does not work
    recon_loss = get_ae_loss_value(y_hat, y_target, class_weights)
    # kl
    kl = kl_divergence(z, mean, std)

    # elbo
    elbo = kl.mean() + recon_loss

    return elbo

def model_step(config, model, b_input, b_target, label_tags, class_weights, io_normalisation_values, autocast_enabled=False):
    b_input = b_input-io_normalisation_values['target_mean'].to(b_input.device)
    b_input = b_input/io_normalisation_values['target_std'].to(b_input.device)

    ### Forward pass ###
    with amp.autocast(enabled=autocast_enabled):
        assert b_input.dim() == 5, \
            f"Input image for model must be {5}D: BxCxSPATIAL but is {b_input.shape}"

        if config.model_type == 'vae':
            y_hat, (z, mean, std) = model(b_input)
        elif config.model_type == 'ae':
            y_hat, _ = model(b_input)
        else:
            raise ValueError
        # Reverse normalisation to outputs
        y_hat = y_hat*io_normalisation_values['target_std'].to(b_input.device)
        y_hat = y_hat+io_normalisation_values['target_mean'].to(b_input.device)

        ### Calculate loss ###
        assert y_hat.dim() == 5, \
            f"Input shape for loss must be {5}D: BxNUM_CLASSESxSPATIAL but is {y_hat.shape}"
        assert b_target.dim() == 5, \
            f"Target shape for loss must be {5}D: BxNUM_CLASSESxSPATIAL but is {b_target.shape}"

        if "vae" in type(model).__name__.lower():
            loss = get_vae_loss_value(y_hat, b_target.float(), z, mean, std, class_weights, model)
        else:
            loss = get_ae_loss_value(y_hat, b_target.float(), class_weights)

    return y_hat, loss



def epoch_iter(global_idx, config, model, dataset, dataloader, class_weights, fold_postfix, phase='train', autocast_enabled=False, optimizer=None, scheduler=None, scaler=None):
    PHASES = ['train', 'val', 'test']
    assert phase in ['train', 'val', 'test'], f"phase must be one of {PHASES}"

    epx_losses = []
    dices = []
    class_dices = []

    if phase == 'train':
        model.train()
        dataset.train(use_modified=False)
    else:
        model.eval()
        dataset.eval()

    for batch_idx, batch in tqdm(enumerate(dataloader), desc=phase, total=len(dataloader)):

        b_input, b_seg = get_model_input(batch, config, len(dataset.label_tags))
        if phase == 'train':
            optimizer.zero_grad()
            y_hat, loss = model_step(config, model, b_input, b_seg, dataset.label_tags, class_weights, dataset.io_normalisation_values, autocast_enabled)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ###  Scheduler management ###
            if config.use_scheduling:
                scheduler.step(loss)

        else:
            with torch.no_grad():
                y_hat, loss = model_step(config, model, b_input, b_seg, dataset.label_tags, class_weights, dataset.io_normalisation_values, autocast_enabled)

        epx_losses.append(loss.item())

        pred_seg = y_hat.argmax(1)

        # Calculate dice score
        b_dice = dice3d(
            torch.nn.functional.one_hot(pred_seg, len(dataset.label_tags)).permute(0,4,1,2,3),
            b_seg,
            one_hot_torch_style=False
        )

        dices.append(get_batch_dice_over_all(
            b_dice, exclude_bg=True))
        class_dices.append(get_batch_dice_per_class(
            b_dice, dataset.label_tags, exclude_bg=True))

        if config.debug: break

    loss_mean = torch.tensor(epx_losses).mean()
    ### Logging ###
    print(f"### {phase.upper()}")

    ### Log wandb data ###
    log_id = f'losses/{phase}_loss{fold_postfix}'
    log_val = loss_mean
    wandb.log({log_id: log_val}, step=global_idx)
    print(f'losses/{phase}_loss{fold_postfix}', f"{log_val}")

    log_id = f'scores/{phase}_dice_mean_wo_bg{fold_postfix}'
    log_val = np.nanmean(dices)
    print(log_id, f"{log_val*100:.2f}%")
    wandb.log({log_id: log_val}, step=global_idx)

    log_class_dices(f"scores/{phase}_dice_mean_", fold_postfix, class_dices, global_idx)
    print()
    print()

    return loss_mean



def run_dl(run_name, config, training_dataset, test_dataset):
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

        best_val_loss = 0
        train_idxs = torch.tensor(train_idxs)
        val_idxs = torch.tensor(val_idxs)
        val_ids = training_dataset.switch_3d_identifiers(val_idxs)

        print(f"Will run validation with these 3D samples (#{len(val_ids)}):", sorted(val_ids))

        ### Add train sampler and dataloaders ##
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idxs)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idxs)
        test_subsampler = torch.utils.data.SubsetRandomSampler(range(len(test_dataset)))

        train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size,
            sampler=train_subsampler, pin_memory=False, drop_last=False,
            # collate_fn=training_dataset.get_efficient_augmentation_collate_fn()
        )
        val_dataloader = DataLoader(training_dataset, batch_size=config.val_batch_size,
            sampler=val_subsampler, pin_memory=False, drop_last=False,
        )
        test_dataloader = DataLoader(test_dataset, batch_size=config.val_batch_size,
            sampler=test_subsampler, pin_memory=False, drop_last=False,
        )

        ### Get model, data parameters, optimizers for model and data parameters, as well as grad scaler ###
        if 'checkpoint_epx' in config and config['checkpoint_epx'] is not None:
            epx_start = config['checkpoint_epx']
        else:
            epx_start = 0

        if config.checkpoint_name:
            # Load from checkpoint
            _path = f"{config.mdl_save_prefix}/{config.checkpoint_name}_fold{fold_idx}_epx{epx_start}"
        else:
            _path = f"{config.mdl_save_prefix}/{wandb.run.name}_fold{fold_idx}_epx{epx_start}"

        (model, optimizer, scheduler, scaler) = get_model(config, len(training_dataset), len(training_dataset.label_tags),
            THIS_SCRIPT_DIR=THIS_SCRIPT_DIR, _path=_path, device=config.device)

        all_bn_counts = torch.zeros([len(training_dataset.label_tags)], device='cpu')

        for bn_counts in training_dataset.bincounts_3d.values():
            all_bn_counts += bn_counts

        class_weights = 1 / (all_bn_counts).float().pow(.35)
        class_weights /= class_weights.mean()

        class_weights = class_weights.to(device=config.device)

        autocast_enabled = 'cuda' in config.device
        autocast_enabled = False

        for epx in range(epx_start, config.epochs):
            global_idx = get_global_idx(fold_idx, epx, config.epochs)
            # Log the epoch idx per fold - so we can recover the diagram by setting
            # ref_epoch_idx as x-axis in wandb interface
            print(f"### Log epoch {epx}")
            wandb.log({"ref_epoch_idx": epx}, step=global_idx)

            _ = epoch_iter(global_idx, config, model, training_dataset, train_dataloader, class_weights, fold_postfix,
                phase='train', autocast_enabled=autocast_enabled, optimizer=optimizer, scheduler=scheduler, scaler=scaler)

            val_loss = epoch_iter(global_idx, config, model, training_dataset, val_dataloader, class_weights, fold_postfix,
                phase='val', autocast_enabled=autocast_enabled, optimizer=None, scheduler=None, scaler=None)

            _ = epoch_iter(global_idx, config, model, test_dataset, test_dataloader, class_weights, fold_postfix,
                phase='test', autocast_enabled=autocast_enabled, optimizer=None, scheduler=None, scaler=None)

            print()

            # Save model
            if config.save_every is None:
                pass

            elif config.save_every == 'best':
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = f"{config.mdl_save_prefix}/{wandb.run.name}{fold_postfix}_best"
                    save_model(
                        Path(THIS_SCRIPT_DIR, save_path),
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler)

            elif (epx % config.save_every == 0) or (epx+1 == config.epochs):
                save_path = f"{config.mdl_save_prefix}/{wandb.run.name}{fold_postfix}_epx{epx}"
                save_model(
                    Path(THIS_SCRIPT_DIR, save_path),
                    model=model,
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

            if config.debug:
                break

        # End of fold loop


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

# %%
# Config overrides
# config_dict['wandb_mode'] = 'disabled'
# config_dict['debug'] = True
# Model loading
# config_dict['checkpoint_name'] = 'ethereal-serenity-1138'
# config_dict['fold_override'] = 0
# config_dict['checkpoint_epx'] = 39

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
        # training_dataset = prepare_data(config_dict)
        config = wandb.config

        run_dl(run_name, config, training_dataset, test_dataset)

def sweep_run():
    with wandb.init() as run:
        run = wandb.init(
            settings=wandb.Settings(start_method="thread"),
            mode=config_dict['wandb_mode']
        )

        run_name = run.name
        print("Running", run_name)
        # training_dataset = prepare_data(config)
        config = wandb.config

        run_dl(run_name, config, training_dataset, test_dataset)

if config_dict['do_sweep']:
    # Integrate all config_dict entries into sweep_dict.parameters -> sweep overrides config_dict
    cp_config_dict = copy.deepcopy(config_dict)
    # cp_config_dict.update(copy.deepcopy(sweep_config_dict['parameters']))
    for del_key in sweep_config_dict['parameters'].keys():
        if del_key in cp_config_dict:
            del cp_config_dict[del_key]
    merged_sweep_config_dict = copy.deepcopy(sweep_config_dict)
    # merged_sweep_config_dict.update(cp_config_dict)
    for key, value in cp_config_dict.items():
        merged_sweep_config_dict['parameters'][key] = dict(value=value)
    # Convert enum values in parameters to string. They will be identified by their numerical index otherwise
    for key, param_dict in merged_sweep_config_dict['parameters'].items():
        if 'value' in param_dict and isinstance(param_dict['value'], Enum):
            param_dict['value'] = str(param_dict['value'])
        if 'values' in param_dict:
            param_dict['values'] = [str(elem) if isinstance(elem, Enum) else elem for elem in param_dict['values']]

        merged_sweep_config_dict['parameters'][key] = param_dict

    sweep_id = wandb.sweep(merged_sweep_config_dict, project=PROJECT_NAME)
    wandb.agent(sweep_id, function=sweep_run)

else:
    normal_run()

# %%
if not in_notebook():
    sys.exit(0)

# %%
# Do any postprocessing / visualization in notebook here

# %%
