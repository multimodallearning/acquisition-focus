import numpy as np
import torch
import nibabel as nib
from pathlib import Path
from torch.utils.checkpoint import checkpoint

import einops as eo

def switch_rows(affine_mat):
    affine_mat[:,:3] = affine_mat[:,:3].flip(1)
    return affine_mat



def get_transformed_affine_from_grid_affine(grid_affine, volume_affine, ras_transform_mat, volume_shape, fov_vox):

    B = grid_affine.shape[0]
    # From torch grid resample affine [-1,+1] get the affine matrix for the transformed nifti volume
    transformed_affine = grid_affine

    # Adjust offset
    transformed_affine[:,:3,-1] = (transformed_affine[:,:3,-1]+1.0) / 2.0*fov_vox

    # Switch back rows
    transformed_affine = switch_rows(transformed_affine)
    transformed_affine = transformed_affine.transpose(2,1)
    transformed_affine = switch_rows(transformed_affine)
    transformed_affine = transformed_affine.transpose(2,1)

    # See get_grid_affine_from_ras_affines, here we divide by the rescaling factor
    rescale_mat = \
        volume_shape.view(1,3) * (1/volume_shape).view(3,1)

    transformed_affine[:,:3,:3] = transformed_affine[:,:3,:3] / rescale_mat

    # Rescale matrix for output field of view
    fov_scale = volume_shape / fov_vox
    fov_scale_mat = torch.eye(4).to(volume_affine)
    fov_scale_mat[:3,:3] = torch.diag(fov_scale)
    transformed_affine = transformed_affine @ fov_scale_mat.view(1,4,4)

    translat_rotat_part = -(transformed_affine[:,:3,:3] @ (fov_vox/2.0).to(volume_affine))
    # translat_rotat_mat = torch.eye(4).view(1,4,4).to(dtype=transformed_affine.dtype, device=transformed_affine.device)
    # translat_rotat_mat = translat_rotat_mat.repeat(4,1,1)
    translat_rotat_mat = torch.zeros_like(transformed_affine)
    translat_rotat_mat[:,:3,-1] = translat_rotat_part
    transformed_affine = volume_affine @ transformed_affine

    transformed_affine[:,:3,-1] = (volume_affine @ translat_rotat_mat)[:,:3,-1] + ras_transform_mat[:,:3,-1]

    return transformed_affine



def get_grid_affine_from_ras_affines(volume_affine, ras_transform_mat, volume_shape, fov_mm, pre_grid_sample_affine):

    # (IJK -> RAS+).inverse() @ (Slice -> RAS+) == Slice -> IJK
    affine_mat = volume_affine.inverse() @ ras_transform_mat

    # Rescale matrix for field of view
    fov_scale = fov_mm / volume_shape
    fov_scale_mat = torch.eye(4).to(volume_affine)
    fov_scale_mat[:3,:3] = torch.diag(fov_scale)
    affine_mat = affine_mat @ fov_scale_mat

    # Adjust offset
    affine_mat[:,:3,-1] = (affine_mat[:,:3,-1])*2.0/volume_shape - 1.0

    # Rescale matrix by D,H,W dimension
    # affine_mat[:3, :3] = torch.tensor([
    #     [mat[0,0]*D/D, mat[0,1]*H/D, mat[0,2]*W/D],
    #     [mat[1,0]*D/H, mat[1,1]*H/H, mat[1,2]*W/H],
    #     [mat[2,0]*D/W, mat[2,1]*H/W, mat[2,2]*W/W]
    # ])
    # Is equivalent to:
    rescale_mat = \
        volume_shape.view(1,3) * (1/volume_shape).view(3,1)

    affine_mat[:,:3,:3] = affine_mat[:,:3,:3] * rescale_mat

    # Switch D,W dimension of matrix (needs two times switching on rows and on columns)
    affine_mat = switch_rows(affine_mat)
    affine_mat = affine_mat.transpose(2,1)
    affine_mat = switch_rows(affine_mat)
    affine_mat = affine_mat.transpose(2,1)

    # Reflect on last dimension (only needed for slicer perfect view alignment, otherwise result is mirrored)
    reflect_mat = torch.tensor([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,-1,0],
        [0,0,0,1]
    ]).to(volume_affine)
    affine_mat = affine_mat @ reflect_mat.view(1,4,4)

    if pre_grid_sample_affine is None:
        return affine_mat

    return affine_mat @ pre_grid_sample_affine

def do_sample(volume, grid, gs_kwargs):
    if volume.requires_grad or grid.requires_grad:
        transformed = checkpoint(
            torch.nn.functional.grid_sample, volume, grid,
            *list(gs_kwargs.values()))
    else:
        transformed = torch.nn.functional.grid_sample(volume, grid, **gs_kwargs)
    return transformed

def nifti_transform(volume:torch.Tensor, volume_affine:torch.Tensor, ras_transform_mat: torch.Tensor, fov_mm, fov_vox,
    is_label=False, pre_grid_sample_affine=None, pre_grid_sample_hidden_affine=None, dtype=torch.float32):

    DIM = volume.dim()
    assert DIM == 5
    B = volume.shape[0]

    if ras_transform_mat is None:
        # This matrix auto-centers the view
        # (i.e. a quasi no-op for transformed voxel array no rotation,
        # but zoom is going on according to fov_mm, fov_vox)
        fov_mm_center = (
            volume_affine @ torch.as_tensor(list(volume.shape[-3:])+ [1.], dtype=volume_affine.dtype)
            + volume_affine @ torch.tensor([0.,0.,0.,1.], dtype=volume_affine.dtype)
        ) / 2
        # ras_transform_mat = torch.eye(4).repeat(B,1,1)
        ras_transform_mat = torch.tensor([
            [1., 0., 0., 0],
            [0., 0., -1., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.]
        ]).repeat(B,1,1)
        ras_transform_mat[:,:3,-1] = fov_mm_center[:,:3].view(1,3)

    assert volume_affine.dim() == ras_transform_mat.dim() == 3 \
        and B == volume_affine.shape[0] \
        and B == ras_transform_mat.shape[0]

    if pre_grid_sample_affine is not None:
        assert pre_grid_sample_affine.dim() == 3 \
            and B == pre_grid_sample_affine.shape[0]
    if pre_grid_sample_hidden_affine is not None:
        assert pre_grid_sample_hidden_affine.dim() == 3 \
            and B == pre_grid_sample_hidden_affine.shape[0]

    device = volume.device
    fov_mm = fov_mm.to(device)
    fov_vox = fov_vox.to(device)
    volume_affine = volume_affine.to(device)
    ras_transform_mat = ras_transform_mat.to(volume_affine)

    if pre_grid_sample_affine is not None:
        pre_grid_sample_affine = pre_grid_sample_affine.to(volume_affine)

    # Prepare volume
    B,C,D,H,W = volume.shape
    volume_shape = torch.tensor([D,H,W], device=device)
    initial_dtype = volume.dtype

    # Get the affine for torch grid resampling from RAS space
    grid_affine = get_grid_affine_from_ras_affines(volume_affine, ras_transform_mat, volume_shape, fov_mm, pre_grid_sample_affine)
    target_shape = torch.Size([B,C] + fov_vox.tolist())

    if pre_grid_sample_hidden_affine is not None:
        pre_grid_sample_hidden_affine = pre_grid_sample_hidden_affine.to(volume_affine)
    else:
        pre_grid_sample_hidden_affine = torch.eye(4).to(volume_affine)

    augmented_grid_affine = (grid_affine @ pre_grid_sample_hidden_affine)

    grid = torch.nn.functional.affine_grid(
        augmented_grid_affine[:,:3,:].view(B,3,4), target_shape, align_corners=False
    ).to(device=volume.device)

    if is_label:
        gs_kwargs = dict(
            mode='nearest',
            padding_mode='zeros',
            align_corners=False
        )
        transformed = do_sample(volume.to(dtype=dtype), grid.to(dtype=dtype), gs_kwargs)

    else:
        gs_kwargs = dict(
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )
        min_value = volume.min()
        volume = volume - min_value
        transformed = do_sample(volume.to(dtype=dtype), grid.to(dtype=dtype), gs_kwargs)
        transformed = transformed + min_value

    transformed = transformed.view(target_shape)

    # Rebuild affine
    grid_affine_cloned = grid_affine.clone().detach()

    # This will only be for nifti output, no gradients here.
    transformed_nii_affine = get_transformed_affine_from_grid_affine(grid_affine_cloned,
        volume_affine, ras_transform_mat, volume_shape, fov_vox)

    return transformed, grid_affine, transformed_nii_affine



def get_crop_affine(affine, vox_offset):
    mm_offset = affine[:,:-1,:-1] @ vox_offset.to(affine)
    affine[:,:-1,-1] = affine[:,:-1,-1] + mm_offset
    return affine




def crop_around_label_center(label: torch.Tensor, vox_size: torch.Tensor, image: torch.Tensor=None,
    affine: torch.Tensor=None, center_mode='mean'):
    """This function crops (or pags) a label and a corresponding image to a specified voxel size
       based on the label centroid position (2D or 3D input).

    Args:
        label (torch.Tensor): Label map to crop
        vox_size (torch.Tensor): The target size, i.e. torch.tensor([200,100,100]).
            A dimension with size -1 will not be cropped.
        image (torch.Tensor): Optional image. Defaults to None.
        affine (torch.Tensor, optional): A nifti affine which is adjusted accordingly. Defaults to None.

    Returns:
        _type_: _description_
    """
    n_dims = label.dim()-2
    assert n_dims == vox_size.numel()
    assert center_mode in ['mean', 'minmax']
    B,C_LAB,*_ = label.shape

    vox_size = vox_size.int().to(device=label.device)
    label_shape = torch.as_tensor(label.shape, device=label.device).int()[2:]
    no_crop = (vox_size == -1)
    # vox_size[no_crop] = label_shape[no_crop]

    sp_idxs = label.long().to_sparse()._indices()

    if center_mode == 'mean':
        label_center = sp_idxs.float().mean(dim=1).int()[-n_dims:]
    elif center_mode == 'minmax':
        label_center = (
            (sp_idxs.float().amin(dim=1)+sp_idxs.float().amax(dim=1))/2
        ).round().int()[-n_dims:]

    # This is the true bounding box in label space when cropping to the demanded size
    in_bbox_min = (label_center-((vox_size+.5)/2.)).round().int()
    in_bbox_max = label_center+(vox_size/2.).round().int()

    in_bbox_min[no_crop] = 0
    in_bbox_max[no_crop] = label_shape[no_crop]

    # This is the 'bounding box' in the output data space (the region of the cropped data we fill)
    out_bbox_min = torch.zeros(n_dims, dtype=torch.int)
    out_bbox_max = vox_size.clone().int()
    out_bbox_max[no_crop] = label_shape[no_crop]

    in_crop_slcs = [slice(None), slice(None)]
    out_crop_slcs = [slice(None), slice(None)]

    cropped_label = torch.zeros([B,C_LAB]+(out_bbox_max-out_bbox_min).tolist(), dtype=torch.int, device=label.device)

    if image is not None:
        _, C_IM, *_ = image.shape
        cropped_image = torch.zeros([B,C_IM]+(out_bbox_max-out_bbox_min).tolist(), dtype=image.dtype, device=image.device)

    in_clip_min = in_bbox_min.clip(min=0)-in_bbox_min
    in_clip_max = in_bbox_max.clip(max=label_shape)-in_bbox_max

    in_bbox_min_clip = in_bbox_min + in_clip_min
    out_bbox_min_clip = out_bbox_min + in_clip_min

    in_bbox_max_clip = in_bbox_max + in_clip_max
    out_bbox_max_clip = out_bbox_max + in_clip_max

    for dim_idx in range(n_dims):
        # Create slices
        in_crop_slcs.append(slice(in_bbox_min_clip[dim_idx], in_bbox_max_clip[dim_idx]))
        out_crop_slcs.append(slice(out_bbox_min_clip[dim_idx], out_bbox_max_clip[dim_idx]))

    # Crop the data

    cropped_label[out_crop_slcs] = label[in_crop_slcs]

    if image is not None:
        cropped_image[out_crop_slcs] = image[in_crop_slcs]
    else:
        cropped_image = None

    if affine is not None:
        # If an affine was passed recalculate the new affine
        vox_offset = torch.tensor([slc.start.item() for slc in in_crop_slcs[-n_dims:]])
        affine = get_crop_affine(affine, vox_offset)

    return cropped_label, cropped_image, affine




def align_to_sa_hla_from_volume(base_dir, volume, initial_affine, align_affine, fov_mm, fov_vox, is_label):
    fov_mm, fov_vox = torch.tensor(fov_mm), torch.tensor(fov_vox)

    # Only grid sample the center slice
    fov_mm_slice = fov_mm.clone()
    fov_mm_slice[-1] /= fov_vox[-1]
    fov_vox_slice = fov_vox.clone()
    fov_vox_slice[-1] = 1

    base_dir = Path(base_dir)
    hla_affine_path = Path(base_dir.parent.parent, "slice_inflate/preprocessing", "mmwhs_1002_4CH.mat")
    sa_affine_path =  Path(base_dir.parent.parent, "slice_inflate/preprocessing", "mmwhs_1002_SA.mat")

    hla_affine = align_affine @ torch.from_numpy(np.loadtxt(hla_affine_path))
    sa_affine =  align_affine @ torch.from_numpy(np.loadtxt(sa_affine_path))



def cut_slice(b_volume):
    b_volume = eo.rearrange(b_volume, 'B C D H W -> W B C D H')
    center_idx = b_volume.shape[0]//2
    b_volume = b_volume[center_idx:center_idx+1]
    return eo.rearrange(b_volume, ' W B C D H -> B C D H W')

def soft_cut_slice(b_volume, std=50.0):
    b_volume = eo.rearrange(b_volume, 'B C D H W -> W B C D H')
    W = b_volume.shape[0]
    center_idx = W//2

    n_dist = torch.distributions.normal.Normal(torch.tensor(center_idx), torch.tensor(std))

    probs = torch.arange(0, W)
    probs = n_dist.log_prob(probs).exp()
    probs = probs / probs.max()
    probs = probs.to(b_volume.device)

    b_volume = (b_volume * probs.view(W,1,1,1,1)).sum(0, keepdim=True)

    return eo.rearrange(b_volume, ' W B C D H -> B C D H W')