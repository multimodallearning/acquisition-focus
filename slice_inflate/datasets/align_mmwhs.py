import numpy as np
import torch
import nibabel as nib
from pathlib import Path



def switch_rows(affine_mat):
    affine_mat[:3] = affine_mat[:3].flip(0)
    return affine_mat



def get_transformed_affine_from_grid_affine(grid_affine, volume_affine, ras_affine_mat, volume_shape, fov_mm, fov_vox):
    # From torch grid resample affine [-1,+1] get the affine matrix for the transformed nifti volume
    transformed_affine = grid_affine.clone()

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
    fov_scale_mat = torch.eye(4).to(dtype=grid_affine.dtype)
    fov_scale_mat[:3,:3] = torch.diag(fov_scale)
    transformed_affine = transformed_affine @ fov_scale_mat

    translat_rotat_part = -(transformed_affine[:3,:3] @ (fov_vox/2.0).to(dtype=grid_affine.dtype))
    transformed_affine = volume_affine @ transformed_affine

    transformed_affine[:3,-1] = volume_affine[:3,:3] @ translat_rotat_part + ras_affine_mat[:3,-1]

    return transformed_affine



def get_grid_affine_from_ras_affines(volume_affine, ras_affine_mat, volume_shape, fov_mm):

    ras_affine_mat = ras_affine_mat.to(dtype=volume_affine.dtype)

    # (IJK -> RAS+).inverse() @ (Slice -> RAS+) == Slice -> IJK
    affine_mat = volume_affine.inverse() @ ras_affine_mat

    # Rescale matrix for field of view
    fov_scale = fov_mm / volume_shape
    fov_scale_mat = torch.eye(4).to(dtype=volume_affine.dtype)
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
    ]).to(dtype=volume_affine.dtype)
    affine_mat = affine_mat @ reflect_mat

    return affine_mat



def nifti_transform(volume:torch.Tensor, volume_affine:torch.Tensor, ras_affine_mat: torch.Tensor, fov_mm, fov_vox,
    is_label=False, dtype=torch.float32):

    # Prepare volume
    B,C,D,H,W = volume.shape
    volume_shape = torch.tensor([D,H,W])
    initial_dtype = volume.dtype

    # Get the affine for torch grid resampling from RAS space
    grid_affine = get_grid_affine_from_ras_affines(volume_affine, ras_affine_mat, volume_shape, fov_mm)

    target_shape = torch.Size([B,C] + fov_vox.tolist())

    grid = torch.nn.functional.affine_grid(
        grid_affine[:3,:].view(1,3,4), target_shape, align_corners=False
    ).to(device=volume.device)

    if is_label:
        transformed = torch.nn.functional.grid_sample(
            volume.to(dtype=dtype), grid.to(dtype=dtype), align_corners=False, mode='nearest'
        )
    else:
        min_value = volume.min()
        volume = volume - min_value
        transformed = torch.nn.functional.grid_sample(
            volume.to(dtype=dtype), grid.to(dtype=dtype), align_corners=False, padding_mode='zeros'
        )
        transformed = transformed + min_value

    transformed = transformed.view(target_shape)

    # Rebuild affine
    transformed_affine = get_transformed_affine_from_grid_affine(grid_affine,
        volume_affine, ras_affine_mat, volume_shape, fov_mm, fov_vox)

    return transformed, transformed_affine



def crop_around_label_center(image, label, vox_size):
    spatial_dims = label.dim()-2
    vox_size = vox_size.to(device=label.device)
    sp_label = label.long().to_sparse()
    sp_idxs = sp_label._indices()[-spatial_dims:]
    lbl_shape = torch.as_tensor(label.shape)
    label_center = sp_idxs.float().mean(dim=1).int()
    bbox_max = label_center+(vox_size/2).int()
    bbox_min = label_center-((vox_size+1)/2).int()

    crop_slcs = [slice(None), slice(None)] # Select all for B,C dimension
    for dim_idx in range(spatial_dims):
        # Check bounds of crop and correct for every spatial dim (not B,C)
        if bbox_min[dim_idx] < 0:
            bbox_min[dim_idx] = 0
            bbox_max[dim_idx] = 0 + vox_size[dim_idx]

        elif bbox_max[dim_idx] > lbl_shape[2+dim_idx]:
            bbox_min[dim_idx] = lbl_shape[2+dim_idx] - vox_size[dim_idx]
            bbox_max[dim_idx] = lbl_shape[2+dim_idx]

        crop_slcs.append(slice(bbox_min[dim_idx], bbox_max[dim_idx]))

    return image[crop_slcs], label[crop_slcs]



def cut_slice(volume):
    return volume[:,:,volume.shape[-1]//2]



def align_to_sa_hla_from_volume(base_dir, volume, initial_affine, sa_align_affine, hla_align_affine, fov_mm, fov_vox, is_label):
    fov_mm, fov_vox = torch.tensor(fov_mm), torch.tensor(fov_vox)

    # Only grid sample the center slice
    fov_mm_slice = fov_mm.clone()
    fov_mm_slice[-1] /= fov_vox[-1]
    fov_vox_slice = fov_vox.clone()
    fov_vox_slice[-1] = 1

    base_dir = Path(base_dir)
    hla_affine_path = Path(base_dir.parent.parent, "slice_inflate/preprocessing", "mmwhs_1002_HLA_red_slice_to_ras.mat")
    sa_affine_path =  Path(base_dir.parent.parent, "slice_inflate/preprocessing", "mmwhs_1002_SA_yellow_slice_to_ras.mat")

    hla_affine = hla_align_affine @ torch.from_numpy(np.loadtxt(hla_affine_path))
    sa_affine =  sa_align_affine @ torch.from_numpy(np.loadtxt(sa_affine_path))

    aligned_sa_volume, aligned_sa_affine = nifti_transform(volume, initial_affine, sa_affine, fov_mm=fov_mm, fov_vox=fov_vox,
        is_label=is_label)

    # # Do only retrieve the center slice for HLA view: Be careful. Output volume is ok, but not hla_affine for 1-vox slice
    # aligned_hla_volume, aligned_hla_affine = nifti_transform(volume, initial_affine, hla_affine, fov_mm=fov_mm_slice, fov_vox=fov_vox_slice,
    #     is_label=is_label)
    aligned_hla_volume, aligned_hla_affine = nifti_transform(volume, initial_affine, hla_affine, fov_mm=fov_mm, fov_vox=fov_vox,
        is_label=is_label)

    return dict(
        aligned_sa_volume=aligned_sa_volume,
        aligned_sa_affine=aligned_sa_affine,
        aligned_hla_volume=aligned_hla_volume,
        aligned_hla_affine=aligned_hla_affine
    )