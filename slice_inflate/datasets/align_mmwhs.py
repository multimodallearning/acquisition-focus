import numpy as np
import torch
import nibabel as nib
from pathlib import Path

FOV_MM = torch.tensor([300,300,300])
FOV_VOX = torch.tensor([200,200,200])



def switch_rows(affine_mat):
    affine_mat = affine_mat.clone()
    affine_mat[:3] = affine_mat.detach()[:3].flip(0)
    return affine_mat



def slicer_slice_transform(nii_volume, ras_affine_mat: np.ndarray, fov_mm, fov_vox, is_label=False):

    # (IJK -> RAS+).inverse() @ (Slice -> RAS+) == Slice -> IJK
    affine_mat =  torch.tensor(nii_volume.affine).inverse() @ ras_affine_mat

    # Get spacing and size related metrics
    nii_shape = torch.tensor(nii_volume.shape)

    origin = torch.tensor(nii_volume.affine[:3,-1])

    fov_scale = fov_mm / nii_shape

    # Prepare volume
    volume = torch.from_numpy(nii_volume.get_fdata())
    volume = volume.view([1,1]+nii_shape.tolist())

    # Rescale matrix for field of view
    fov_scale_mat = torch.eye(4)
    fov_scale_mat[:3,:3] = torch.diag(fov_scale)
    affine_mat = affine_mat @ fov_scale_mat.double()

    # Adjust offset
    affine_mat[:3,-1] = (affine_mat[:3,-1])*2.0/nii_shape - 1.0 # double origin? # double spacing?

    # Rescale matrix by D,H,W dimension
    # affine_mat[:3, :3] = torch.tensor([
    #     [mat[0,0]*D/D, mat[0,1]*H/D, mat[0,2]*W/D],
    #     [mat[1,0]*D/H, mat[1,1]*H/H, mat[1,2]*W/H],
    #     [mat[2,0]*D/W, mat[2,1]*H/W, mat[2,2]*W/W]
    # ])
    # Is equivalent to:
    rescale_mat = \
        nii_shape.view(1,3) * (1/nii_shape).view(3,1) # double spacing?

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
    ]).double()
    affine_mat = affine_mat @ reflect_mat

    target_shape = torch.Size([1, 1] + fov_vox.tolist())

    grid = torch.nn.functional.affine_grid(affine_mat[:3,:].view(1,3,4), target_shape, align_corners=False)

    if is_label:
        resampled = torch.nn.functional.grid_sample(volume, grid, align_corners=False, mode='nearest')
    else:
        resampled = torch.nn.functional.grid_sample(volume, grid, align_corners=False)

    resampled = resampled.squeeze()

    resampled_affine = torch.eye(4)
    resampled_affine[:3,:3] = torch.diag(fov_mm/fov_vox) # TODO: This is not 100% correct - volume is not aligned correctly in space but at least undistorted

    nii_resampled = nib.Nifti1Image(resampled.cpu().numpy(), affine=resampled_affine)
    return nii_resampled



def crop_around_label_center(image, label, vox_size):
    n_dims = label.dim()
    sp_label = label.to_sparse()
    sp_idxs = sp_label._indices()
    lbl_shape = torch.as_tensor(label.shape)
    label_center = \
        torch.div((sp_idxs.max(dim=1)[0] + sp_idxs.min(dim=1)[0]), 2).int()
    bbox_max = label_center+((vox_size)/2).int()
    bbox_min = label_center-((vox_size+1)/2).int()

    crop_slcs = []
    for dim_idx in range(n_dims):
        # Check bounds of crop and correct
        if bbox_min[dim_idx] < 0:
            bbox_min[dim_idx] = 0
            bbox_max[dim_idx] = 0 + vox_size[dim_idx]

        elif bbox_max[dim_idx] > lbl_shape[dim_idx]:
            bbox_min[dim_idx] = lbl_shape[dim_idx] - vox_size[dim_idx]
            bbox_max[dim_idx] = lbl_shape[dim_idx]

        crop_slcs.append(slice(bbox_min[dim_idx], bbox_max[dim_idx]))

    return image[crop_slcs], label[crop_slcs]



def cut_slice(nii_volume):
    return torch.as_tensor(nii_volume.get_fdata()[:,:,nii_volume.shape[-1]//2])



def align_global(base_dir, nii_path, _3d_id, is_label):
    base_dir = Path(base_dir)
    nii_volume = nib.load(nii_path)
    align_mat_path = Path(base_dir, "preprocessed", f"f1002mr_m{_3d_id.split('-')[0]}{_3d_id.split('-')[1]}.mat")
    hla_mat_path = Path(base_dir.parent.parent, "slice_inflate/preprocessing", "mmwhs_1002_HLA_red_slice_to_ras.mat")

    align_mat = torch.from_numpy(np.loadtxt(align_mat_path))
    hla_mat = align_mat @ torch.from_numpy(np.loadtxt(hla_mat_path))

    aligned_nii_volume = slicer_slice_transform(nii_volume, hla_mat, fov_mm=FOV_MM, fov_vox=FOV_VOX, is_label=is_label)
    return aligned_nii_volume



def cut_sa_hla_slice_from_volume(base_dir, nii_path, _3d_id, is_label):
    base_dir = Path(base_dir)
    nii_volume = nib.load(nii_path)

    align_mat_path = Path(base_dir, "preprocessed", f"f1002mr_m{_3d_id.split('-')[0]}{_3d_id.split('-')[1]}.mat")
    hla_mat_path = Path(base_dir.parent.parent, "slice_inflate/preprocessing", "mmwhs_1002_HLA_red_slice_to_ras.mat")
    sa_mat_path =  Path(base_dir.parent.parent, "slice_inflate/preprocessing", "mmwhs_1002_SA_yellow_slice_to_ras.mat")

    align_mat = torch.from_numpy(np.loadtxt(align_mat_path))
    hla_mat = align_mat @ torch.from_numpy(np.loadtxt(hla_mat_path))
    sa_mat =  align_mat @ torch.from_numpy(np.loadtxt(sa_mat_path))

    aligned_hla_nii_volume = slicer_slice_transform(nii_volume, hla_mat, fov_mm=FOV_MM, fov_vox=FOV_VOX, is_label=is_label)
    aligned_sa_nii_volume = slicer_slice_transform(nii_volume, sa_mat, fov_mm=FOV_MM, fov_vox=FOV_VOX, is_label=is_label)

    hla_slc = cut_slice(aligned_hla_nii_volume)
    sa_slc = cut_slice(aligned_sa_nii_volume)

    return hla_slc, sa_slc