# def cut_slice(volume):
#     return torch.from_numpy(volume.get_fdata())[:,:,volume.shape[-1]//2]

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

# def crop_to_label_center(image_path, label_path, vox_size):

#     label_center = !c3d {label_path} -centroid
#     label_center_vox = label_center[0]

#     str_elems = re.sub(r"[\[\]A-Za-z\,_]", "", label_center_vox).strip().split(' ')

#     vox_center = np.array([float(elem) for elem in str_elems])
#     vox_lower = np.floor(vox_center - vox_size/2).clip(min=0)
#     v_origin = 'x'.join([str(int(elem)) for elem in vox_lower.tolist()])+'vox'
#     v_size = 'x'.join([str(int(elem)) for elem in vox_size.tolist()])+'vox'

#     !c3d {image_path} -region {v_origin} {v_size} -o {image_path}
#     !c3d {label_path} -region {v_origin} {v_size} -o {label_path}