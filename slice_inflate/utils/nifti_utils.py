import torch
import torch.cuda.amp as amp
from torch.utils.checkpoint import checkpoint



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
    scale_mat = torch.zeros_like(affine)
    scale_mat[:,:3,:3] = torch.diag_embed(scaler)
    scale_mat[:,-1,-1] = 1.
    affine = affine @ scale_mat
    return affine



def get_grid_affine_and_nii_affine(
    volume_nii_affine, ras_transform_affine, fov_vox_i, target_fov_mm, target_fov_vox, pre_grid_sample_affine
):
    # (IJK -> RAS+).inverse() @ (Slice -> RAS+) == Slice -> IJK
    affine_mat = volume_nii_affine.inverse() @ ras_transform_affine

    # Calculate preliminary vars
    fov_mm_i = get_zooms(volume_nii_affine) * fov_vox_i
    zooms_i = get_zooms(volume_nii_affine)


    # Adjust offset and switch D,W dimension of matrix (needs two times switching on rows and on columns)
    affine_mat[:,:3,-1] = get_torch_translation_from_pix_translation(affine_mat[:,:3,-1], fov_vox_i)
    affine_mat = switch_0_2_mat_dim(affine_mat)

    # Apply external grid_sample affine
    affine_mat =  affine_mat @ pre_grid_sample_affine

    # Adjust zooms
    affine_mat = rescale_rot_components_with_diag(
        affine_mat,
        (1/get_zooms(affine_mat) * (target_fov_mm / fov_mm_i)).flip(1)
    )

    # Now get Nifti-matrix
    nii_affine = affine_mat.clone()
    nii_affine = switch_0_2_mat_dim(nii_affine)
    nii_affine = rescale_rot_components_with_diag(nii_affine,
        fov_mm_i / (target_fov_vox * zooms_i)
    )
    nii_affine[:,:3,-1] = get_pix_translation_from_torch_translation(nii_affine[:,:3,-1], fov_vox_i)
    neg_half_mm_shift = volume_nii_affine[:,:3,:3] @ nii_affine[:,:3,:3] @ (-(target_fov_vox-1)/2.0).to(volume_nii_affine)

    nii_affine = volume_nii_affine @ nii_affine # Pix to mm space
    nii_affine[:,:3,-1] += neg_half_mm_shift
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



def get_noop_ras_transfrom_mat(volume_nii_affine, fov_vox_i):
    B = volume_nii_affine.shape[0]
    # This RAS matrix will not change the pixel orientations, nor the resulting nifti affine
    # (i.e. a quasi no-op for transformed voxel array no rotation,
    # but zoom is going on according to target_fov_mm, target_fov_vox)
    fov_vox_center = torch.as_tensor(list(fov_vox_i)+ [1.]).to(volume_nii_affine) / 2.
    ras_transform_affine = torch.eye(4)[None].repeat(B,1,1).to(volume_nii_affine)
    ras_transform_affine[:,:3,-1] += fov_vox_center[:3]

    noop_mat = volume_nii_affine @ ras_transform_affine
    return noop_mat



def nifti_grid_sample(volume:torch.Tensor, volume_nii_affine:torch.Tensor,
                      ras_transform_affine:torch.Tensor=None,
                      target_fov_mm:torch.Tensor=None, target_fov_vox:torch.Tensor=None,
                      is_label:bool=False, pre_grid_sample_affine:torch.Tensor=None, dtype=torch.float32):
    # Transforms a given volume and its corresponding nifti affine matrix with torch.nn.functional.grid_sample
    # and preserves the physical orientation by adjusting the nifti affine matrix
    #
    # volume: The input volume to be transformed. Usually extracted by torch.as_tensor(nibabel.load(path).get_fdata())
    # volume_nii_affine: The corresponding nifti matrix. Usually extracted by torch.as_tensor(nibabel.load(path).affine)
    # target_fov_mm: Float tensor of three elements containing the target spacing. E.g. torch.tensor([1.5,1.5,1.5])
    # target_fov_vox: Long tensor of three elements containing the target voxel size. E.g. torch.tensor([100,120,130])
    # is_label: Switch to enable interpolation based or nearest sampling
    # pre_grid_sample_affine: A torch space grid sample affine matrix that is applied to the voxel data
    # dtype: Datatype that is used internally in the sampling operation

    # Works with nibabel loaded nii(.gz) files, ITK loading untested
    DIM = volume.dim()
    assert DIM == 5

    assert isinstance(volume, torch.Tensor) and isinstance(volume_nii_affine, torch.Tensor)
    if pre_grid_sample_affine is not None:
        assert isinstance(pre_grid_sample_affine, torch.Tensor)

    device = volume.device
    initial_dtype = volume.dtype
    B,C,D,H,W = volume.shape
    fov_vox_i = torch.tensor([D,H,W]).to(device).double()

    if target_fov_mm is None:
        target_fov_mm = get_zooms(volume_nii_affine) * fov_vox_i
    if target_fov_vox is None:
        target_fov_vox = torch.as_tensor(volume.shape[-3:])

    target_shape = torch.Size([B,C] + target_fov_vox.tolist())

    if pre_grid_sample_affine is not None:
        assert pre_grid_sample_affine.dim() == 3 \
            and B == pre_grid_sample_affine.shape[0]

    target_fov_mm = target_fov_mm.to(device).to(device).double()
    target_fov_vox = target_fov_vox.to(device).to(device).double()
    volume_nii_affine = volume_nii_affine.to(device).double()

    if ras_transform_affine is None:
        ras_transform_affine = get_noop_ras_transfrom_mat(volume_nii_affine, fov_vox_i)
    else:
        raise Warning("Providing a RAS space transform matrix is experimental and might produce wrong results.")

    ras_transform_affine = ras_transform_affine.to(device).double()

    assert volume_nii_affine.dim() == ras_transform_affine.dim() == 3 \
        and B == volume_nii_affine.shape[0] \
        and B == ras_transform_affine.shape[0]

    if pre_grid_sample_affine is None:
        pre_grid_sample_affine = torch.eye(4)[None]
    pre_grid_sample_affine = pre_grid_sample_affine.to(device).double()

    # Get affines
    grid_affine, transformed_nii_affine = get_grid_affine_and_nii_affine(
        volume_nii_affine, ras_transform_affine, fov_vox_i, target_fov_mm, target_fov_vox, pre_grid_sample_affine
    )

    if 'int' in str(initial_dtype):
        volume = volume.to(dtype=dtype)
        grid_affine = grid_affine.to(dtype=dtype)
    else:
        grid_affine = grid_affine.to(volume)

    with amp.autocast(enabled=False):
        grid = torch.nn.functional.affine_grid(
            grid_affine[:,:3,:].view(B,3,4), target_shape, align_corners=False
        )

    if is_label:
        gs_kwargs = dict(
            mode='nearest',
            padding_mode='zeros',
            align_corners=False
        )
        transformed = do_sample(volume, grid, gs_kwargs)

    else:
        gs_kwargs = dict(
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        min_value = volume.min()
        volume = volume - min_value
        transformed = do_sample(volume, grid, gs_kwargs)
        transformed = transformed + min_value

    transformed = transformed.view(target_shape).to(dtype=initial_dtype)

    return transformed, grid_affine, transformed_nii_affine



def crop_around_label_center(label: torch.Tensor, volume_nii_affine: torch.Tensor,
                              target_fov_mm: torch.Tensor, target_fov_vox: torch.Tensor, image: torch.Tensor=None,
                              center_mode='mean'):

    n_dims = label.dim()-2
    assert n_dims == target_fov_vox.numel()
    assert center_mode in ['mean', 'minmax']

    target_fov_vox = target_fov_vox.int().to(device=label.device)

    label_shape = torch.as_tensor(label.shape, device=label.device).int()[2:]
    no_crop = (target_fov_vox == -1)
    target_fov_vox[no_crop] = label_shape[no_crop]

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
        cropped_image, *_ = nifti_grid_sample(image, volume_nii_affine, ras_transform_affine=None, target_fov_mm=target_fov_mm, target_fov_vox=target_fov_vox,
            is_label=False, pre_grid_sample_affine=pre_grid_sample_affine,
            # pre_grid_sample_hidden_affine=None,
            dtype=torch.float32)
    else:
        cropped_image = None

    cropped_label, _, cropped_nii_affine = nifti_grid_sample(label, volume_nii_affine, ras_transform_affine=None, target_fov_mm=target_fov_mm, target_fov_vox=target_fov_vox,
            is_label=True, pre_grid_sample_affine=pre_grid_sample_affine,
            # pre_grid_sample_hidden_affine=None,
            dtype=torch.float32)

    return cropped_label, cropped_image, cropped_nii_affine



def get_zooms(nii_affine):
    assert nii_affine.dim() == 3
    return (nii_affine[:,:3,:3]*nii_affine[:,:3,:3]).sum(1).sqrt()