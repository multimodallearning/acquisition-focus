import numpy as np
import torch
import nibabel as nib
from pathlib import Path
from torch.utils.checkpoint import checkpoint

import einops as eo

def switch_rows(affine_mat):
    affine_mat[:3] = affine_mat[:3].flip(0)
    return affine_mat



def get_transformed_affine_from_grid_affine(grid_affine, volume_affine, ras_affine_mat, volume_shape, fov_vox):
    # From torch grid resample affine [-1,+1] get the affine matrix for the transformed nifti volume
    transformed_affine = grid_affine

    # Adjust offset
    transformed_affine[:3,-1] = (transformed_affine[:3,-1]+1.0) / 2.0*fov_vox

    # Switch back rows
    transformed_affine = switch_rows(transformed_affine)
    transformed_affine = transformed_affine.T
    transformed_affine = switch_rows(transformed_affine)
    transformed_affine = transformed_affine.T

    # See get_grid_affine_from_ras_affines, here we divide by the rescaling factor
    rescale_mat = \
        volume_shape.view(1,3) * (1/volume_shape).view(3,1)

    transformed_affine[:3, :3] = transformed_affine[:3, :3] / rescale_mat

    # Rescale matrix for output field of view
    fov_scale = volume_shape / fov_vox
    fov_scale_mat = torch.eye(4).to(volume_affine)
    fov_scale_mat[:3,:3] = torch.diag(fov_scale)
    transformed_affine = transformed_affine @ fov_scale_mat

    translat_rotat_part = -(transformed_affine[:3,:3] @ (fov_vox/2.0).to(volume_affine))
    transformed_affine = volume_affine @ transformed_affine

    transformed_affine[:3,-1] = volume_affine[:3,:3] @ translat_rotat_part + ras_affine_mat[:3,-1]

    return transformed_affine



def get_grid_affine_from_ras_affines(volume_affine, ras_affine_mat, volume_shape, fov_mm):

    # (IJK -> RAS+).inverse() @ (Slice -> RAS+) == Slice -> IJK
    affine_mat = volume_affine.inverse() @ ras_affine_mat

    # Rescale matrix for field of view
    fov_scale = fov_mm / volume_shape
    fov_scale_mat = torch.eye(4).to(volume_affine)
    fov_scale_mat[:3,:3] = torch.diag(fov_scale)
    affine_mat = affine_mat @ fov_scale_mat

    # Adjust offset
    affine_mat[:3,-1] = (affine_mat[:3,-1])*2.0/volume_shape - 1.0

    # Rescale matrix by D,H,W dimension
    # affine_mat[:3, :3] = torch.tensor([
    #     [mat[0,0]*D/D, mat[0,1]*H/D, mat[0,2]*W/D],
    #     [mat[1,0]*D/H, mat[1,1]*H/H, mat[1,2]*W/H],
    #     [mat[2,0]*D/W, mat[2,1]*H/W, mat[2,2]*W/W]
    # ])
    # Is equivalent to:
    rescale_mat = \
        volume_shape.view(1,3) * (1/volume_shape).view(3,1)

    affine_mat[:3, :3] = affine_mat[:3, :3] * rescale_mat

    # Switch D,W dimension of matrix (needs two times switching on rows and on columns)
    affine_mat = switch_rows(affine_mat)
    affine_mat = affine_mat.T
    affine_mat = switch_rows(affine_mat)
    affine_mat = affine_mat.T

    # Reflect on last dimension (only needed for slicer perfect view alignment, otherwise result is mirrored)
    reflect_mat = torch.tensor([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,-1,0],
        [0,0,0,1]
    ]).to(volume_affine)
    affine_mat = affine_mat @ reflect_mat

    return affine_mat



def nifti_transform(volume:torch.Tensor, volume_affine:torch.Tensor, ras_affine_mat: torch.Tensor, fov_mm, fov_vox,
    is_label=False, dtype=torch.float32):

    device = volume.device
    fov_mm = fov_mm.to(device)
    fov_vox = fov_vox.to(device)
    volume_affine = volume_affine.to(device)
    ras_affine_mat = ras_affine_mat.to(volume_affine)

    # Prepare volume
    B,C,D,H,W = volume.shape
    volume_shape = torch.tensor([D,H,W], device=device)
    initial_dtype = volume.dtype

    # Get the affine for torch grid resampling from RAS space
    grid_affine = get_grid_affine_from_ras_affines(volume_affine, ras_affine_mat, volume_shape, fov_mm)

    target_shape = torch.Size([B,C] + fov_vox.tolist())

    grid = torch.nn.functional.affine_grid(
        grid_affine[:3,:].view(1,3,4), target_shape, align_corners=False
    ).to(device=volume.device)

    if is_label:
        transformed = checkpoint(
            torch.nn.functional.grid_sample,
            volume.to(dtype=dtype), grid.to(dtype=dtype), 'nearest', 'zeros', False
        )
    else:
        min_value = volume.min()
        volume = volume - min_value
        transformed = checkpoint(torch.nn.functional.grid_sample,
            volume.to(dtype=dtype), grid.to(dtype=dtype), 'bilinear', 'zeros', False
        )
        transformed = transformed + min_value

    transformed = transformed.view(target_shape)

    # Rebuild affine
    transformed_affine = get_transformed_affine_from_grid_affine(grid_affine,
        volume_affine, ras_affine_mat, volume_shape, fov_vox)

    return transformed, transformed_affine



def crop_around_label_center(b_image, b_label, vox_size):
    assert b_label.dim() == 5
    if not b_image is None:
        assert b_image.dim() == 5

def get_crop_affine(affine, vox_offset):
    mm_offset = affine[:-1,:-1] @ vox_offset.to(affine)
    crop_affine = affine.clone()
    crop_affine[:-1,-1] = affine[:-1,-1] + mm_offset
    return crop_affine



def crop_around_label_center(label: torch.Tensor, vox_size: torch.Tensor, image: torch.Tensor=None,
    affine: torch.Tensor=None):
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
    B,C_LAB,*_ = label.shape

    vox_size = vox_size.int().to(device=label.device)
    label_shape = torch.as_tensor(label.shape, device=label.device).int()[2:]
    no_crop = (vox_size == -1)
    vox_size[no_crop] = label_shape[no_crop]

    sp_idxs = label.long().to_sparse()._indices()
    label_center = sp_idxs.float().mean(dim=1).int()[-n_dims:]

    # This is the true bounding box in label space when cropping to the demanded size
    in_bbox_min = (label_center-((vox_size+1)/2)+0.5).int()
    in_bbox_max = (label_center+(vox_size/2)+0.5).int()

    # This is the 'bounding box' in the output data space (the region of the cropped data we fill)
    out_bbox_min = torch.zeros(n_dims, dtype=torch.int)
    out_bbox_max = vox_size.clone().int()
    out_bbox_max[no_crop] = label_shape[no_crop]

    in_crop_slcs = [slice(None), slice(None)]
    out_crop_slcs = [slice(None), slice(None)]

    cropped_label = torch.zeros([B,C_LAB]+vox_size.tolist(), dtype=torch.int, device=label.device)

    for dim_idx in range(n_dims):
        # Check bounds of crop and correct
        if in_bbox_min[dim_idx] < 0:
            out_bbox_min[dim_idx] = out_bbox_min[dim_idx]-in_bbox_min[dim_idx]
            in_bbox_min[dim_idx] = 0

        elif in_bbox_max[dim_idx] > label_shape[dim_idx]:
            out_bbox_max[dim_idx] = out_bbox_max[dim_idx]-in_bbox_max[dim_idx]
            in_bbox_max[dim_idx] = label_shape[dim_idx]

        in_crop_slcs.append(slice(in_bbox_min[dim_idx], in_bbox_max[dim_idx]))
        out_crop_slcs.append(slice(out_bbox_min[dim_idx], out_bbox_max[dim_idx]))

    # Crop the data
    if image is not None:
        _, C_IM, *_ = image.shape
        cropped_image = torch.zeros([B,C_IM]+vox_size.tolist(), dtype=image.dtype, device=image.device)
        cropped_image[out_crop_slcs] = image[in_crop_slcs]
    else:
        cropped_image = None

    cropped_label[out_crop_slcs] = label[in_crop_slcs]

    if affine is not None:
        # If an affine was passed recalculate the new affine
        vox_offset = torch.tensor([slc.start.item() for slc in in_crop_slcs[-n_dims:]])
        affine = get_crop_affine(affine, vox_offset)

    return cropped_label, cropped_image, affine



def cut_slice(volume):
    return volume[:,:,volume.shape[-1]//2]



def align_to_sa_hla_from_volume(base_dir, volume, initial_affine, align_affine, fov_mm, fov_vox, is_label):
    fov_mm, fov_vox = torch.tensor(fov_mm), torch.tensor(fov_vox)

    # Only grid sample the center slice
    fov_mm_slice = fov_mm.clone()
    fov_mm_slice[-1] /= fov_vox[-1]
    fov_vox_slice = fov_vox.clone()
    fov_vox_slice[-1] = 1

    base_dir = Path(base_dir)
    hla_affine_path = Path(base_dir.parent.parent, "slice_inflate/preprocessing", "mmwhs_1002_HLA_red_slice_to_ras.mat")
    sa_affine_path =  Path(base_dir.parent.parent, "slice_inflate/preprocessing", "mmwhs_1002_SA_yellow_slice_to_ras.mat")

    hla_affine = align_affine @ torch.from_numpy(np.loadtxt(hla_affine_path))
    sa_affine =  align_affine @ torch.from_numpy(np.loadtxt(sa_affine_path))



def cut_slice(b_volume):
    b_volume = eo.rearrange(b_volume, 'B C D H W -> W B C D H')
    center_idx = b_volume.shape[0]//2
    b_volume = b_volume[center_idx:center_idx+1]
    return eo.rearrange(b_volume, ' W B C D H -> B C D H W')