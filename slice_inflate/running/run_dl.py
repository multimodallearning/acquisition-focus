import os
from pathlib import Path
import contextlib
from collections import defaultdict

import dill
from tqdm import tqdm
import wandb

from pytorch_run_on_recommended_gpu.run_on_recommended_gpu import get_cuda_environ_vars as get_vars
os.environ.update(get_vars(os.environ.get('CUDA_VISIBLE_DEVICES','0')))
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
import einops as eo
import numpy as np
import monai
# import nibabel as nib

from slice_inflate.utils.nnunetv2_utils import DC_and_CE_loss
from slice_inflate.models.interface_models import EPix2Vox_InterfaceModel
from slice_inflate.models.learnable_transform import ATModulesContainer, AffineTransformModule, get_random_affine
from slice_inflate.models.hybrid_unet import HybridUnet
from slice_inflate.utils.log_utils import get_global_idx, log_label_metrics, \
    log_oa_metrics, log_affine_param_stats, log_frameless_image, get_cuda_mem_info_str
from slice_inflate.functional.clinical_cardiac_views import get_class_volumes
from slice_inflate.utils.torch_utils import get_batch_score_per_label, save_model, \
    reduce_label_scores_epoch, get_binarized_from_onehot_label, NoneOptimizer, set_requires_grad
from slice_inflate.utils.nifti_utils import nifti_grid_sample, get_zooms
from slice_inflate.related_works.epix2vox.epix2vox import EPix2VoxModel128, get_optimizer_and_scheduler
# torch.autograd.set_detect_anomaly(True)



def get_reconstruction_model(config, num_classes, save_path=None, load_model_only=False):
    device = config.device
    assert config.model_type in ['hybrid-unet', 'hybrid-EPix2Vox', 'hybrid-Pix2Vox']

    n_views = len(config.view_ids)
    if config.model_type == 'hybrid-unet':
        model = HybridUnet(n_views=n_views, num_classes=num_classes)

    elif config.model_type == 'hybrid-EPix2Vox':
        epix_model = EPix2VoxModel128(
            use_merger=True, use_refiner=True, n_views=n_views, use_epix2vox=True,
            epoch_start_use_merger=0, epoch_start_use_refiner=0
        )
        model = EPix2Vox_InterfaceModel(epix_model)

    elif config.model_type == 'hybrid-Pix2Vox':
        epix_model = EPix2VoxModel128(
            use_merger=True, use_refiner=True, n_views=n_views, use_epix2vox=False,
            epoch_start_use_merger=0, epoch_start_use_refiner=0
        )
        model = EPix2Vox_InterfaceModel(epix_model)

    else:
        raise ValueError

    model.to(device)

    if save_path:
        assert Path(save_path).is_dir()
        model_dict = torch.load(Path(save_path).joinpath('model.pth'), map_location=device)
        epx = model_dict.get('metadata', {}).get('epx', 0)
        print(f"Loading model from {save_path}")
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

    scaler = torch.cuda.amp.GradScaler()

    if save_path and not load_model_only:
        assert Path(save_path).is_dir()
        print(f"Loading optimizer, scheduler, scaler from {save_path}")
        optimizer.load_state_dict(torch.load(Path(save_path).joinpath('optimizer.pth'), map_location=device))
        scheduler.load_state_dict(torch.load(Path(save_path).joinpath('scheduler.pth'), map_location=device))
        scaler.load_state_dict(torch.load(Path(save_path).joinpath('scaler.pth'), map_location=device))

    else:
        print(f"Generated fresh optimizer, scheduler, scaler.")

    # for submodule in model.modules():
    #     submodule.register_forward_hook(anomaly_hook)

    return (model, optimizer, scheduler, scaler), epx



def get_transform_model(config, num_classes, save_path=None):
    device = config.device

    atm_container = ATModulesContainer(config, num_classes)
    atm_container.to(device)
    set_requires_grad(atm_container, False)

    if config.view_optimization_mode == 'opt-current-fix-previous':
        set_requires_grad([atm_container.get_next_non_optimized_view_module().localization_net], True)
    elif config.view_optimization_mode == 'opt-all':
        set_requires_grad([mod.localization_net for mod in atm_container], True)
    elif config.view_optimization_mode == 'opt-none':
        pass
    else:
        raise ValueError()

    transform_parameters = atm_container.get_next_non_optimized_view_module().parameters()

    if config.view_optimization_mode in ['opt-all', 'opt-current-fix-previous']:
        assert config.use_affine_theta
        transform_optimizer = torch.optim.AdamW(transform_parameters, weight_decay=0.1, lr=config.lr*2.0)
        transform_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(transform_optimizer, T_0=int(config.epochs/4)+1)
    else:
        transform_optimizer = NoneOptimizer()
        transform_scheduler = None

    if save_path and Path(save_path).is_dir():
        print(f"Loading transform optimizer from {save_path}")
        atm_container.load_state_dict(torch.load(Path(save_path)/ 'atm_container.pth',map_location=device), strict=False)
        transform_optimizer.load_state_dict(torch.load(Path(save_path).joinpath('transform_optimizer.pth'), map_location=device))

    else:
        print(f"Generated fresh transform optimizer.")

    return atm_container, transform_optimizer, transform_scheduler



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



def get_input_affine_for_atm(atm, base_affine, b_view_affines):
    B = base_affine.shape[0]

    if atm.view_id == 'RND':
        return atm.random_grid_affine.repeat(B,1,1)

    return base_affine.inverse() \
            @ torch.as_tensor(b_view_affines[atm.view_id]).view(B,4,4).to(base_affine)



def get_reconstruction_model_input(batch, phase, config, num_classes, atm_container, segment_fn):
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
            target_fov_mm=torch.as_tensor(config.hires_fov_mm), target_fov_vox=torch.as_tensor(config.hires_fov_vox),
            is_label=True, pre_grid_sample_affine=base_affine)
        b_image, _, _ = nifti_grid_sample(b_image.unsqueeze(1), nifti_affine,
            target_fov_mm=torch.as_tensor(config.hires_fov_mm), target_fov_vox=torch.as_tensor(config.hires_fov_vox),
            is_label=False, pre_grid_sample_affine=base_affine)
        b_label = b_label.squeeze(1)
        b_image = b_image.squeeze(1)

    b_label = eo.rearrange(F.one_hot(b_label, num_classes),
                        'B D H W OH -> B OH D H W')
    B,NUM_CLASSES,D,H,W = b_label.shape
    b_soft_label = b_label.float()

    for atm in atm_container:
        atm.use_affine_theta = config.use_affine_theta

    # Get the initial view orientations (w/o optimized theta, this is added later)
    active_view_modules = [mod for mod in atm_container.get_active_view_modules()]
    input_grid_affines = [get_input_affine_for_atm(mod, base_affine, b_view_affines).to(nifti_affine)\
                          for mod in active_view_modules]

    if config.do_augment_input_orientation and phase in config.aug_phases:
        input_grid_affines = apply_affine_augmentation(input_grid_affines,
            rotation_strength=.1*config.sample_augment_strength,
            zoom_strength=.2*config.sample_augment_strength,
            offset_strength=0.,
        )

    label_slices, output_grid_affines = [], []

    for atm, input_ga in zip(active_view_modules, input_grid_affines):
        if config.view_optimization_mode == 'opt-current-fix-previous' and atm == active_view_modules[-1]:
            ctx = contextlib.nullcontext # With grad
        elif config.view_optimization_mode == 'opt-all':
            ctx = contextlib.nullcontext # With grad
        else:
            ctx = torch.no_grad

        with ctx():
            image_slc, label_slc, output_ga = \
                get_transformed(
                    config,
                    phase,
                    b_label.view(B, NUM_CLASSES, D, H, W),
                    b_soft_label.view(B, NUM_CLASSES, D, H, W),
                    nifti_affine,
                    input_ga,
                    atm,
                    image=b_image.view(B, 1, D, H, W), segment_fn=segment_fn)

            # Now apply augmentation that adds uncertainty to the inverse resconstruction grid sampling
            if config.do_augment_recon_orientation and phase in config.aug_phases:
                output_ga = apply_affine_augmentation([output_ga],
                    rotation_strength=.1*config.sample_augment_strength,
                    zoom_strength=.2*config.sample_augment_strength,
                    offset_strength=0.,
                )[0].to(nifti_affine)

            label_slices.append(label_slc)
            output_grid_affines.append(output_ga)

            # from matplotlib import pyplot as plt
            # plt.imshow(label_slc[0].argmax(0).squeeze().cpu().numpy())
            # plt.savefig('label_slc.png')

    n_views = len(config.view_ids)
    n_active_views = len(active_view_modules)

    # Fill views if not all are present (currently optimized view will be duplicated)
    slices = label_slices + [label_slices[-1]] * (n_views-n_active_views)
    grid_affines = output_grid_affines + [output_grid_affines[-1]] * (n_views-n_active_views)

    b_input = torch.cat(slices, dim=1).squeeze(-1)
    assert b_input.dim() == 4

    b_target = b_label
    return b_input, b_target, grid_affines



def get_loss_value(y_hat, y_target):
    return DC_and_CE_loss({}, {})(y_hat, y_target.argmax(1, keepdim=True))



def model_step(config, phase, epx, model, atm_container, batch, label_tags, segment_fn, autocast_enabled=False):

    ### Forward pass ###
    with amp.autocast(enabled=autocast_enabled):
        b_input, b_target, b_grid_affines = get_reconstruction_model_input(batch, phase, config, len(label_tags), atm_container, segment_fn)

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



def epoch_iter(epx, global_idx, config, model, atm_container, dataset, dataloader, phase='train',
    autocast_enabled=False, all_optimizers=None, scaler=None, store_net_output_to=None):
    PHASES = ['train', 'val', 'test']
    assert phase in ['train', 'val', 'test'], f"phase must be one of {PHASES}"

    epx_losses = []

    epx_theta_grid_affines = defaultdict(lambda: {})
    epx_transformed_nifti_affines = defaultdict(lambda: {})

    epx_input = {}

    label_scores_epoch = {}
    seg_metrics_nanmean = {}
    seg_metrics_std = {}
    seg_metrics_nanmean_oa = {}
    seg_metrics_std_oa = {}

    if phase == 'train':
        model.train()

        if config.train_affine_theta:
            atm_container.eval()
            if 'opt-current-fix-previous' in config.view_optimization_mode:
                atm_container.get_next_non_optimized_view_module().train()

    else:
        model.eval()
        atm_container.eval()

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
                model, atm_container,
                batch,
                dataset.label_tags, segment_fn, autocast_enabled)

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
                    model, atm_container,
                    batch,
                    dataset.label_tags, segment_fn, autocast_enabled)

            epx_losses.append(loss.item())

        epx_input.update({k:v for k,v in zip(batch['id'], b_input.cpu())})

        for atm in atm_container.get_active_view_modules():
            # Store affines for file output
            lga_batch_dict = {k:v for k,v in zip(batch['id'], atm.last_grid_affine.cpu())}
            epx_theta_grid_affines[atm.view_id].update(lga_batch_dict)

            ltna_batch_dict = {k:v for k,v in zip(batch['id'], atm.last_transformed_nifti_affine.cpu())}
            epx_transformed_nifti_affines[atm.view_id].update(ltna_batch_dict)

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

    if config.do_output:
        affine_dct = dict(
            epx_theta_grid_affines=epx_theta_grid_affines,
            epx_transformed_nifti_affines=epx_transformed_nifti_affines,
        )
        with open(output_dir/f"affines_{phase}_epx_{epx:05d}.pth", 'wb') as f:
            dill.dump(affine_dct, f)

    if config.do_output and epx_input:
        # Store the slice model input
        save_input = torch.stack(list(epx_input.values()))
        n_views = len(config.view_ids)

        save_input = save_input.chunk(n_views, dim=1)
        save_input = torch.cat([slc.argmax(1, keepdim=True) for slc in save_input], dim=1)

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

    return loss_mean



def run_dl(base_dir, config, fold_properties, stage=None, training_dataset=None, test_dataset=None, run_test_once_only=False):
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
    (model, optimizer, scheduler, scaler), epx_start = get_reconstruction_model(
        config, len(training_dataset.label_tags),
        save_path=mdl_chk_path, load_model_only=False)

    # Load transformation model from checkpoint, if any
    transform_mdl_chk_path = config.transform_model_checkpoint_path if 'transform_model_checkpoint_path' in config else None

    atm_container, transform_optimizer, transform_scheduler = get_transform_model(
        config, len(training_dataset.label_tags), save_path=transform_mdl_chk_path)

    all_optimizers = dict(optimizer=optimizer, transform_optimizer=transform_optimizer)
    all_schedulers = dict(scheduler=scheduler, transform_scheduler=transform_scheduler)

    autocast_enabled = 'cuda' in config.device and config['use_autocast']

    # c_model = torch.compile(model)

    for epx in range(epx_start, config.epochs):
        global_idx = get_global_idx(fold_idx, epx, config.epochs)
        # Log the epoch idx per fold - so we can recover the diagram by setting
        # ref_epoch_idx as x-axis in wandb interface
        print( f"### Log epoch {epx}/{config.epochs}")
        wandb.log({"ref_epoch_idx": epx}, step=global_idx)

        if not run_test_once_only:
            train_loss = epoch_iter(
                epx, global_idx, config, model, atm_container,
                training_dataset, train_dataloader,
                phase='train', autocast_enabled=autocast_enabled,
                all_optimizers=all_optimizers, scaler=scaler, store_net_output_to=None)

            val_loss, _ = epoch_iter(epx, global_idx, config, model, atm_container, training_dataset, val_dataloader,
                phase='val', autocast_enabled=autocast_enabled, all_optimizers=None, scaler=None, store_net_output_to=None)

        test_loss, _ = epoch_iter(epx, global_idx, config, model, atm_container, test_dataset, test_dataloader,
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
                save_path = Path(base_dir, save_path)

                if stage is not None:
                    stage['save_path'] = save_path

                save_model(
                    save_path,
                    epx=epx,
                    loss=train_loss,
                    model=model,
                    atm_container=atm_container,
                    optimizer=all_optimizers['optimizer'],
                    transform_optimizer=all_optimizers['transform_optimizer'],
                    scheduler=scheduler,
                    scaler=scaler)

        elif (epx % config.save_every == 0) or (epx+1 == config.epochs):
            save_path = f"{config.mdl_save_prefix}/{wandb.run.name}_epx{epx}"
            save_path = Path(base_dir, save_path)

            if stage is not None:
                stage['save_path'] = save_path

            save_model(
                save_path,
                epx=epx,
                loss=train_loss,
                model=model,
                atm_container=atm_container,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler)

        # End of epoch loop

        if config.debug or run_test_once_only:
            break