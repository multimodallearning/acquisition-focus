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
from slice_inflate.datasets.mmwhs_dataset import MMWHSDataset, load_data, extract_2d_data
from slice_inflate.utils.common_utils import DotDict, get_script_dir

THIS_SCRIPT_DIR = get_script_dir()
# %%

config_dict = DotDict({
    'num_folds': 5,
    'only_first_fold': True,                # If true do not contiue with training after the first fold
    # 'fold_override': 0,
    # 'checkpoint_epx': 0,

    'use_mind': False,                      # If true use MIND features (https://pubmed.ncbi.nlm.nih.gov/22722056/)
    'epochs': 40,

    'batch_size': 8,
    'val_batch_size': 1,
    'use_2d_normal_to': 'HLA/SA',               # Can be None or 'D', 'H', 'W'. If not None 2D slices will be selected for training

    'atlas_count': 1,                       # If three (noisy) labels per image are used specify three

    'dataset': 'mmwhs',                 # The dataset prepared with our preprocessing scripts
    'data_base_path': str(Path(THIS_SCRIPT_DIR, "data/MMWHS")),
    'reg_state': None, # Registered (noisy) labels used in training. See prepare_data() for valid reg_states
    'train_set_max_len': None,              # Length to cut of dataloader sample count
    'crop_around_3d_label_center': (128,128,128),
    'crop_3d_region': ((0,128), (0,128), (0,128)),        # dimension range in which 3D samples are cropped
    'crop_2d_slices_gt_num_threshold': 0,   # Drop 2D slices if less than threshold pixels are positive

    'lr': 0.01,
    'use_scheduling': True,

    'save_every': 200,
    'mdl_save_prefix': 'data/models',

    'debug': True,
    'wandb_mode': 'online',                         # e.g. online, disabled. Use weights and biases online logging
    'do_sweep': True,                                # Run multiple trainings with varying config values defined in sweep_config_dict below

    # For a snapshot file: dummy-a2p2z76CxhCtwLJApfe8xD_fold0_epx0
    'checkpoint_name': None,                          # Training snapshot name, e.g. dummy-a2p2z76CxhCtwLJApfe8xD
    'fold_override': None,                            # Training fold, e.g. 0
    'checkpoint_epx': None,                           # Training epx, e.g. 0

    'do_plot': False,                                 # Generate plots (debugging purpose)
    'save_dp_figures': False,                         # Plot data parameter value distribution
    'save_labels': True,                              # Store training labels alongside data parameter values inside the training snapshot

    'device': 'cuda'
})

config = config_dict

training_dataset = MMWHSDataset(
    config.data_base_path,
    state="training",
    load_func=load_data,
    extract_slice_func=extract_2d_data,
    modality='mr',
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

# %%
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from slice_inflate.datasets.align_mmwhs import cut_slice



# %%
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
        cut_slice(sample['image'].cpu()),
        cut_slice(sample['label'].cpu()),

        sample['sa_image_slc'].cpu(),
        sample['sa_label_slc'].cpu(),

        sample['hla_image_slc'].cpu(),
        sample['hla_label_slc'].cpu(),
    ]

    for ax, im in zip(grid, show_row):
        ax.imshow(im, cmap='gray', interpolation='none')

    plt.show()

# %%
training_dataset.train()
import nibabel as nib
training_dataset.self_attributes['augment_angle_std'] = 10
print(training_dataset.do_augment)
import torch
lbl, sa_label, hla_label = torch.zeros(128,128), torch.zeros(128,128), torch.zeros(128,128)
for idx in range(15):
    sample = training_dataset[1]
    nib.save(nib.Nifti1Image(sample['label'].cpu().numpy(), affine=torch.eye(4).numpy()), f'out{idx}.nii.gz')
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
training_dataset.train(augment=False)
training_dataset.self_attributes['augment_angle_std'] = 2
print(training_dataset.do_augment)
import torch
lbl, sa_label, hla_label = torch.zeros(128,128), torch.zeros(128,128), torch.zeros(128,128)
for _ in range(5):
    sample = training_dataset[1]

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
