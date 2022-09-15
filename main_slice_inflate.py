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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

from tqdm import tqdm
import wandb
import nibabel as nib

from slice_inflate.datasets.mmwhs_dataset import MMWHSDataset, load_data, extract_2d_data
from slice_inflate.utils.common_utils import DotDict, get_script_dir
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
# %%

config_dict = DotDict({
    'num_folds': 5,
    'only_first_fold': True,                # If true do not contiue with training after the first fold
    # 'fold_override': 0,
    # 'checkpoint_epx': 0,

    'use_mind': False,                      # If true use MIND features (https://pubmed.ncbi.nlm.nih.gov/22722056/)
    'epochs': 100,

    'batch_size': 4,
    'val_batch_size': 1,
    'modality': 'mr',
    'use_2d_normal_to': None,               # Can be None or 'D', 'H', 'W'. If not None 2D slices will be selected for training

    'dataset': 'mmwhs',                 # The dataset prepared with our preprocessing scripts
    'data_base_path': str(Path(THIS_SCRIPT_DIR, "data/MMWHS")),
    'reg_state': None, # Registered (noisy) labels used in training. See prepare_data() for valid reg_states
    'train_set_max_len': None,              # Length to cut of dataloader sample count
    'crop_around_3d_label_center': (128,128,128),
    'crop_3d_region': ((0,128), (0,128), (0,128)),        # dimension range in which 3D samples are cropped
    'crop_2d_slices_gt_num_threshold': 0,   # Drop 2D slices if less than threshold pixels are positive

    'lr': 0.001,
    'use_scheduling': True,

    'save_every': 'best',
    'mdl_save_prefix': 'data/models',

    'debug': False,
    'wandb_mode': 'disabled',                         # e.g. online, disabled. Use weights and biases online logging
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
        state="training",
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
        crop_around_2d_label_center=(128,128),

        augment_angle_std=5,

        device=config.device,
        debug=config.debug
    )

    return training_dataset


# %%
if False:
    training_dataset = prepare_data(config_dict)
    training_dataset.train(augment=False)
    training_dataset.self_attributes['augment_angle_std'] = 2
    print(training_dataset.do_augment)
    for sample in [training_dataset[idx] for idx in [1]]:
        pass
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
    training_dataset = prepare_data(config_dict)
    training_dataset.train()

    training_dataset.self_attributes['augment_angle_std'] = 10
    print(training_dataset.do_augment)
    import torch
    lbl, sa_label, hla_label = torch.zeros(128,128), torch.zeros(128,128), torch.zeros(128,128)
    for idx in range(15):
        sample = training_dataset[1]
        # nib.save(nib.Nifti1Image(sample['label'].cpu().numpy(), affine=torch.eye(4).numpy()), f'out{idx}.nii.gz')
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
    training_dataset = prepare_data(config_dict)
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
        self.third_layer_decoder = self.ConvBlock(40, out_channels_list=[20], strides_list=[1])

        self.fourth_layer_encoder = self.ConvBlock(40, out_channels_list=[60,60,60], strides_list=[2,1,1])
        self.fourth_layer_decoder = self.ConvBlock(decoder_in_channels, out_channels_list=[40], strides_list=[1])

        self.deepest_layer = self.ConvBlock(60, out_channels_list=[60,20,2], strides_list=[2,1,1])

        self.encoder = torch.nn.Sequential(
            self.first_layer_encoder,
            self.second_layer_encoder,
            self.third_layer_encoder,
            self.fourth_layer_encoder,
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            self.fourth_layer_decoder,
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
        return h
        # if self.debug_mode:
        #     return debug_forward_pass(self.encoder, x, STEP_MODE=False)
        # else:
        #     return self.encoder(x)

    def decode(self, z):
        if self.debug_mode:
            return debug_forward_pass(self.decoder, z, STEP_MODE=False)
        else:
            return self.decoder(z)

    def forward(self, x):
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

    def sample_z(self, mean, std):
        return torch.normal(mean=mean, std=std)

    def encode(self, x):
        h = self.encoder(x)
        mean = self.deepest_layer[0](h)
        log_var = self.deepest_layer[1](h)
        return mean, log_var

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.sample_z(mean=mean, std=torch.exp(logvar/2))
        return self.decode(z), z



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
training_dataset = prepare_data(config_dict)


# %%
def nan_hook(self, inp, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print("In", self.__class__.__name__)
            raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

def get_model(config, dataset_len, num_classes, THIS_SCRIPT_DIR, _path=None, device='cpu'):
    _path = Path(THIS_SCRIPT_DIR).joinpath(_path).resolve()

    model = BlendowskiVAE(in_channels=num_classes, out_channels=num_classes)

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

    for submodule in model.modules():
        submodule.register_forward_hook(nan_hook)
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

    return b_input, b_seg

def inference_wrap(model, seg):
    with torch.inference_mode():
        b_seg = seg.unsqueeze(0).unsqueeze(0).float()
        b_out = model(b_seg)[0]
        b_out = b_out.argmax(1)
        return b_out



def train_DL(run_name, config, training_dataset):
    reset_determinism()

    # Configure folds
    kf = KFold(n_splits=config.num_folds)
    # kf.get_n_splits(training_dataset.__len__(use_2d_override=False))
    fold_iter = enumerate(kf.split(range(training_dataset.__len__(use_2d_override=False))))

    if config.get('fold_override', None):
        selected_fold = config.get('fold_override', 0)
        fold_iter = list(fold_iter)[selected_fold:selected_fold+1]
    elif config.only_first_fold:
        fold_iter = list(fold_iter)[0:1]

    if config.use_2d_normal_to is not None:
        n_dims = (-2,-1)
    else:
        n_dims = (-3,-2,-1)

    fold_means_no_bg = []

    best_val_score = 0

    for fold_idx, (train_idxs, val_idxs) in fold_iter:
        train_idxs = torch.tensor(train_idxs)
        val_idxs = torch.tensor(val_idxs)
        val_ids = training_dataset.switch_3d_identifiers(val_idxs)

        print(f"Will run validation with these 3D samples (#{len(val_ids)}):", sorted(val_ids))

        ### Add train sampler and dataloaders ##
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idxs)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idxs)

        train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size,
            sampler=train_subsampler, pin_memory=False, drop_last=False,
            # collate_fn=training_dataset.get_efficient_augmentation_collate_fn()
        )
        val_dataloader = DataLoader(training_dataset, batch_size=config.val_batch_size,
            sampler=val_subsampler, pin_memory=False, drop_last=False,
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

        for epx in range(epx_start, config.epochs):
            global_idx = get_global_idx(fold_idx, epx, config.epochs)

            model.train()

            ### Disturb samples ###
            training_dataset.train(use_modified=False)

            epx_losses = []
            dices = []
            class_dices = []

            # Load data
            for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="batch:", total=len(train_dataloader)):

                optimizer.zero_grad()

                b_input, b_seg = get_model_input(batch, config, len(training_dataset.label_tags))

                ### Forward pass ###
                with amp.autocast(enabled=autocast_enabled):
                    assert b_input.dim() == len(n_dims)+2, \
                        f"Input image for model must be {len(n_dims)+2}D: BxCxSPATIAL but is {b_input.shape}"
                    for param in model.parameters():
                        param.requires_grad = True

                    model.use_checkpointing = True
                    logits = model(b_input)[0]

                    ### Calculate loss ###
                    assert logits.dim() == len(n_dims)+2, \
                        f"Input shape for loss must be BxNUM_CLASSESxSPATIAL but is {logits.shape}"
                    assert b_seg.dim() == len(n_dims)+1, \
                        f"Target shape for loss must be BxSPATIAL but is {b_seg.shape}"

                    ce_loss = nn.CrossEntropyLoss(class_weights)(logits, b_seg)

                    if torch.any(torch.isnan(ce_loss)):
                        raise RuntimeError("NaNs detetected.")

                    scaler.scale(ce_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epx_losses.append(ce_loss.item())

                logits_for_score = logits.argmax(1)

                # Calculate dice score
                b_dice = dice3d(
                    torch.nn.functional.one_hot(logits_for_score, len(training_dataset.label_tags)),
                    torch.nn.functional.one_hot(b_seg, len(training_dataset.label_tags)), # Calculate dice score with original segmentation (no disturbance)
                    one_hot_torch_style=True
                )

                dices.append(get_batch_dice_over_all(
                    b_dice, exclude_bg=True))
                class_dices.append(get_batch_dice_per_class(
                    b_dice, training_dataset.label_tags, exclude_bg=True))


                if config.debug:
                    break

            ###  Scheduler management ###
            if config.use_scheduling:
                scheduler.step(ce_loss)

            ### Logging ###
            print(f"### Log epoch {epx}")
            print("### Training")

            ### Log wandb data ###
            # Log the epoch idx per fold - so we can recover the diagram by setting
            # ref_epoch_idx as x-axis in wandb interface
            wandb.log({"ref_epoch_idx": epx}, step=global_idx)

            mean_loss = torch.tensor(epx_losses).mean()
            wandb.log({f'losses/loss_fold{fold_idx}': mean_loss}, step=global_idx)
            print(f'losses/loss_fold{fold_idx}', f"{mean_loss}")

            mean_dice = np.nanmean(dices)
            print(f'dice_mean_wo_bg_fold{fold_idx}', f"{mean_dice*100:.2f}%")
            wandb.log({f'scores/dice_mean_wo_bg_fold{fold_idx}': mean_dice}, step=global_idx)

            log_class_dices("scores/dice_mean_", f"_fold{fold_idx}", class_dices, global_idx)

            print()
            print("### Validation")
            model.eval()
            training_dataset.eval()

            val_dices = []
            val_class_dices = []

            with amp.autocast(enabled=autocast_enabled):
                with torch.no_grad():
                    for val_batch_idx, val_batch in tqdm(enumerate(val_dataloader), desc="batch:", total=len(val_dataloader)):

                        b_val_input, b_val_seg = get_model_input(val_batch, config, len(training_dataset.label_tags))

                        output_val = model(b_val_input)[0]
                        val_logits_for_score = output_val.argmax(1)

                        b_val_dice = dice3d(
                            torch.nn.functional.one_hot(val_logits_for_score, len(training_dataset.label_tags)),
                            torch.nn.functional.one_hot(b_val_seg, len(training_dataset.label_tags)),
                            one_hot_torch_style=True
                        )

                        # Get mean score over batch
                        val_dices.append(get_batch_dice_over_all(
                            b_val_dice, exclude_bg=True))

                        val_class_dices.append(get_batch_dice_per_class(
                            b_val_dice, training_dataset.label_tags, exclude_bg=True))

                    mean_val_dice = np.nanmean(val_dices)

                    print(f'val_dice_mean_wo_bg_fold{fold_idx}', f"{mean_val_dice*100:.2f}%")
                    wandb.log({f'scores/val_dice_mean_wo_bg_fold{fold_idx}': mean_val_dice}, step=global_idx)
                    log_class_dices("scores/val_dice_mean_", f"_fold{fold_idx}", val_class_dices, global_idx)

            print()

            # Save model
            if config.save_every is None:
                pass

            elif config.save_every == 'best':
                if mean_val_dice > best_val_score:
                    best_val_score = mean_val_dice
                    save_path = f"{config.mdl_save_prefix}/{wandb.run.name}_fold{fold_idx}_best"
                    save_model(
                        Path(THIS_SCRIPT_DIR, save_path),
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler)

            elif (epx % config.save_every == 0) or (epx+1 == config.epochs):
                save_path = f"{config.mdl_save_prefix}/{wandb.run.name}_fold{fold_idx}_epx{epx}"
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

            # End of training loop

            if config.debug:
                break

        # End of fold loop


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

        train_DL(run_name, config, training_dataset)

def sweep_run():
    with wandb.init() as run:
        run = wandb.init(
            settings=wandb.Settings(start_method="thread"),
            mode=config_dict['wandb_mode']
        )

        run_name = run.name
        print("Running", run_name)
        training_dataset = prepare_data(config)
        config = wandb.config

        train_DL(run_name, config, training_dataset)

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
