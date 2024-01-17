import torch
from torch.utils.checkpoint import checkpoint
import einops as eo


def flip_mat_rows_old(affine_mat):
    affine_mat[:,:3] = affine_mat[:,:3].flip(1)
    return affine_mat



def switch_0_2_mat_dim_old(affine_mat):
    affine_mat = flip_mat_rows_old(affine_mat.clone())
    affine_mat = affine_mat.transpose(2,1)
    affine_mat = flip_mat_rows_old(affine_mat)
    return affine_mat.transpose(2,1)



def flip_mat_cols_0_2(affine_mat):
    _,R,C = affine_mat.shape
    flip_affine = torch.tensor([
        [0., 0., 1., 0.],
        [0., 1., 0., 0.],
        [1., 0., 0., 0.],
        [0., 0., 0., 1.]
    ]).view(1,4,4)[:,:R,:C].to(affine_mat)
    return affine_mat @ flip_affine



def switch_0_2_mat_dim(affine_mat):
    affine_mat = flip_mat_cols_0_2(affine_mat)
    affine_mat = affine_mat.transpose(2,1)
    affine_mat = flip_mat_cols_0_2(affine_mat)
    return affine_mat.transpose(2,1)



def rescale_rot_components_with_diag(affine, scaler):
    scale_mat = torch.eye(4)[None].to(affine)
    scale_mat[:,:3,:3] = torch.diag(scaler)
    affine = affine @ scale_mat
    return affine



def rescale_rot_components_with_shape_distortion(affine, volume_shape):
    # Rescale matrix by D,H,W dimension
    # affine_mat[:3, :3] = torch.tensor([
    #     [mat[0,0]*D/D, mat[0,1]*H/D, mat[0,2]*W/D],
    #     [mat[1,0]*D/H, mat[1,1]*H/H, mat[1,2]*W/H],
    #     [mat[2,0]*D/W, mat[2,1]*H/W, mat[2,2]*W/W]
    # ])
    # Is equivalent to:
    rescale_mat = \
        volume_shape.view(1,3) * (1/volume_shape).view(3,1)
    affine[:,:3,:3] = affine[:,:3,:3] * rescale_mat[None]
    return affine



def extract_rot_matrix(affine_mat):
    extractor = torch.eye(4).view(1,4,4).to(affine_mat)
    offset = torch.zeros_like(extractor)
    offset[:,-1,-1] = 1.
    return (affine_mat @ (extractor-offset)) + offset



def extract_translation_matrix(affine_mat):
    extractor = torch.zeros_like(affine_mat)
    extractor[:,-1,-1] = 1.
    offset = torch.eye(4).view(1,4,4).to(affine_mat)
    return affine_mat @ extractor + (offset-extractor)



def extract_translation_comp_only_matrix(affine_mat):
    extractor = torch.zeros_like(affine_mat)
    extractor[:,-1,-1] = 1.
    offset = torch.eye(4).view(1,4,4).to(affine_mat)
    return affine_mat @ extractor - extractor



def get_grid_affine_and_nii_affine(
    volume_affine, ras_transform_mat, volume_shape, fov_mm, fov_vox, pre_grid_sample_affine
):
    B = volume_affine.shape[0]
    pre_grid_sample_affine_rot = extract_rot_matrix(pre_grid_sample_affine) # should be ok...
    pre_grid_sample_affine_translation = pre_grid_sample_affine[:,:3,-1]

    # (IJK -> RAS+).inverse() @ (Slice -> RAS+) == Slice -> IJK
    affine_mat = volume_affine.inverse() @ ras_transform_mat
    affine_mat = (
        switch_0_2_mat_dim(pre_grid_sample_affine_rot) @ extract_rot_matrix(affine_mat)
        + extract_translation_comp_only_matrix(affine_mat)
    )

    # Rescale matrix for field of view
    affine_mat = rescale_rot_components_with_diag(affine_mat, fov_mm / volume_shape)
    # Save pix space matrix for nifti matrix generation
    affine_mat_pix_space = affine_mat.clone()

    # Adjust shape distortions for torch (torch space is always -1;+1 and not D,H,W)
    affine_mat = rescale_rot_components_with_shape_distortion(affine_mat, volume_shape)
    # Adjust offset and switch D,W dimension of matrix (needs two times switching on rows and on columns)
    affine_mat[:,:3,-1] = get_torch_translation_from_pix_translation(affine_mat[:,:3,-1], volume_shape)
    affine_mat = switch_0_2_mat_dim(affine_mat)

    affine_mat[:,:3,-1] = affine_mat[:,:3,-1] + pre_grid_sample_affine_translation

    # Now get Nifti-matrix
    nii_affine = affine_mat_pix_space
    nii_affine = rescale_rot_components_with_diag(nii_affine, volume_shape / fov_vox)

    # TODO there is still a slight offset
    neg_half_mm_shift = volume_affine[:,:3,:3] @ nii_affine[:,:3,:3] @ (-(fov_vox)/2.0).to(volume_affine)

    nii_affine = volume_affine @ nii_affine # Pix to mm space

    translation_offset = (volume_shape.view(1,3) * pre_grid_sample_affine_translation.flip(1)/2).view(B,3)
    translation_offset = (volume_affine[:,:3,:3] @ translation_offset.view(B,3,1)).view(B,3)

    nii_affine[:,:3,-1] += neg_half_mm_shift # To be tested
    nii_affine[:,:3,-1] += translation_offset
    return affine_mat, nii_affine



def get_pix_translation_from_torch_translation(tt, shape_3d):
    pt = (tt+1.0) / 2.0*shape_3d
    return pt



def get_torch_translation_from_pix_translation(pt, shape_3d):
    tt = (pt)*2.0/shape_3d - 1.0
    return tt



def do_sample(volume, grid, gs_kwargs):
    if volume.requires_grad or grid.requires_grad:
        transformed = checkpoint(
            torch.nn.functional.grid_sample, volume, grid,
            *list(gs_kwargs.values()))
    else:
        transformed = torch.nn.functional.grid_sample(volume, grid, **gs_kwargs)
    return transformed



def get_noop_ras_transfrom_mat(volume_affine, volume_shape):
    B = volume_affine.shape[0]
    # This RAS matrix will not change the pixel orientations, nor the resulting nifti affine
    # (i.e. a quasi no-op for transformed voxel array no rotation,
    # but zoom is going on according to fov_mm, fov_vox)
    fov_mm_center = (
        volume_affine @ torch.as_tensor(list(volume_shape)+ [1.]).to(volume_affine)
        + volume_affine @ torch.tensor([0.,0.,0.,1.]).to(volume_affine)
    ) / 2

    ras_transform_mat = torch.tensor([
        [1., 0., 0., 0],
        [0., -1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ]).unsqueeze(0).repeat(B,1,1).to(volume_affine)

    ras_transform_mat[:,:3,-1] = fov_mm_center[:,:3].view(B,3)

    return ras_transform_mat



def nifti_grid_sample(volume:torch.Tensor, volume_affine:torch.Tensor, ras_transform_mat:torch.Tensor=None, fov_mm=None, fov_vox=None,
    is_label=False, pre_grid_sample_affine=None, pre_grid_sample_hidden_affine=None, dtype=torch.float32):
    # Works with nibabel loaded nii(.gz) files, itk loading untested
    assert isinstance(volume, torch.Tensor) and isinstance(volume_affine, torch.Tensor)
    if pre_grid_sample_affine is not None:
        assert isinstance(pre_grid_sample_affine, torch.Tensor)
    if pre_grid_sample_hidden_affine is not None:
        assert isinstance(pre_grid_sample_hidden_affine, torch.Tensor)

    DIM = volume.dim()
    assert DIM == 5

    device = volume.device
    B = volume.shape[0]

    # Prepare shapes
    B,C,D,H,W = volume.shape
    volume_shape = torch.tensor([D,H,W])
    target_shape = torch.Size([B,C] + fov_vox.tolist())

    if pre_grid_sample_affine is not None:
        assert pre_grid_sample_affine.dim() == 3 \
            and B == pre_grid_sample_affine.shape[0]
    if pre_grid_sample_hidden_affine is not None:
        assert pre_grid_sample_hidden_affine.dim() == 3 \
            and B == pre_grid_sample_hidden_affine.shape[0]

    fov_mm = fov_mm.to(dtype=dtype, device=device)
    fov_vox = fov_vox.to(dtype=dtype, device=device)
    volume_affine = volume_affine.to(dtype=dtype, device=device)
    volume_shape = volume_shape.to(dtype=dtype, device=device)

    if ras_transform_mat is None:
        ras_transform_mat = get_noop_ras_transfrom_mat(volume_affine, volume_shape)

    ras_transform_mat = ras_transform_mat.to(volume_affine)
    
    assert volume_affine.dim() == ras_transform_mat.dim() == 3 \
        and B == volume_affine.shape[0] \
        and B == ras_transform_mat.shape[0]


    if pre_grid_sample_affine is None:
        pre_grid_sample_affine = torch.eye(4)[None]
    pre_grid_sample_affine = pre_grid_sample_affine.to(volume_affine)

    if pre_grid_sample_hidden_affine is None:
        pre_grid_sample_hidden_affine = torch.eye(4)[None]
    pre_grid_sample_hidden_affine = pre_grid_sample_hidden_affine.to(volume_affine)

    # Get affines
    grid_affine, transformed_nii_affine = get_grid_affine_and_nii_affine(
        volume_affine, ras_transform_mat, volume_shape, fov_mm, fov_vox, pre_grid_sample_affine
    )
    augmented_grid_affine = (grid_affine @ pre_grid_sample_hidden_affine)

    grid = torch.nn.functional.affine_grid(
        augmented_grid_affine.to(dtype)[:,:3,:].view(B,3,4), target_shape, align_corners=False
    )

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
            padding_mode='zeros',
            align_corners=False
        )
        min_value = volume.min()
        volume = volume - min_value
        transformed = do_sample(volume.to(dtype=dtype), grid.to(dtype=dtype), gs_kwargs)
        transformed = transformed + min_value

    transformed = transformed.view(target_shape)

    return transformed, grid_affine, transformed_nii_affine



def crop_around_label_center(label: torch.Tensor, volume_affine: torch.Tensor,
                              fov_mm: torch.Tensor, fov_vox: torch.Tensor, image: torch.Tensor=None,
                              center_mode='mean'):

    n_dims = label.dim()-2
    assert n_dims == fov_vox.numel()
    assert center_mode in ['mean', 'minmax']

    fov_vox = fov_vox.int().to(device=label.device)

    label_shape = torch.as_tensor(label.shape, device=label.device).int()[2:]
    no_crop = (fov_vox == -1)
    fov_vox[no_crop] = label_shape[no_crop]

    sp_idxs = label.int().to_sparse()._indices()

    if center_mode == 'mean':
        label_center = sp_idxs.float().mean(dim=1).int()[-n_dims:]
    elif center_mode == 'minmax':
        label_center = (
            (sp_idxs.float().amin(dim=1)+sp_idxs.float().amax(dim=1))/2
        ).round().int()[-n_dims:]

    pre_grid_sample_affine = torch.eye(4)[None]
    pre_grid_sample_affine[:,:3,-1] = get_torch_translation_from_pix_translation(label_center, label_shape).flip(0)

    if image is not None:
        cropped_image, *_ = nifti_grid_sample(image, volume_affine, ras_transform_mat=None, fov_mm=fov_mm, fov_vox=fov_vox,
            is_label=False, pre_grid_sample_affine=pre_grid_sample_affine, pre_grid_sample_hidden_affine=None,
            dtype=torch.float32)
    else:
        cropped_image = None

    cropped_label, _, cropped_nii_affine = nifti_grid_sample(label, volume_affine, ras_transform_mat=None, fov_mm=fov_mm, fov_vox=fov_vox,
            is_label=True, pre_grid_sample_affine=pre_grid_sample_affine, pre_grid_sample_hidden_affine=None,
            dtype=torch.float32)

    return cropped_label, cropped_image, cropped_nii_affine


def get_zooms(nii_affine):
    assert nii_affine.dim() == 3
    return (nii_affine[:,:3,:3]*nii_affine[:,:3,:3]).sum(1).sqrt()