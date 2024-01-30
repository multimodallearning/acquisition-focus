import nibabel as nib
import torch
from slice_inflate.datasets.clinical_cardiac_views import get_center_and_median, get_sub_sp_tensor
from slice_inflate.models.learnable_transform import compute_rotation_matrix_from_ortho6d



def convert_centers_to_torch(centers, shp):
    centers = torch.stack(centers).flip(1)
    centers = centers / torch.as_tensor(shp) * 2. - 1.
    centers = torch.cat([centers, torch.ones(centers.shape[0],1)], dim=1)
    return centers



def register_centroids(fixed_label, moving_label, DOF=6):
    assert fixed_label.dtype in [torch.int64, torch.int32]
    assert moving_label.dtype in [torch.int64, torch.int32]
    assert DOF in [6,7]
    assert fixed_label.shape == moving_label.shape
    shp = fixed_label.shape
    assert shp[0] == shp[1] == shp[2]

    fixed_centers = []
    moving_centers = []

    common_classes = set(fixed_label.unique().tolist()).intersection(set(moving_label.unique().tolist()))
    common_classes = list(common_classes)[1:]

    for eq in common_classes:
        fixed_sub = get_sub_sp_tensor(fixed_label.to_sparse(), eq_value=(eq,))
        moving_sub = get_sub_sp_tensor(moving_label.to_sparse(), eq_value=(eq,))
        fixed_cntr = get_center_and_median(fixed_sub)[0]
        moving_cntr = get_center_and_median(moving_sub)[0]
        fixed_centers.append(fixed_cntr)
        moving_centers.append(moving_cntr)

    fixed_centers = convert_centers_to_torch(fixed_centers, shp)
    moving_centers = convert_centers_to_torch(moving_centers, shp)

    if DOF == 7:
        zoom_param = torch.nn.Parameter(torch.randn(1))
    else:
        zoom_param = torch.ones(1)

    rot_params = torch.nn.Parameter(torch.randn(6))
    trans_params = torch.nn.Parameter(torch.randn(3))

    iters = 500

    optim = torch.optim.AdamW([zoom_param, rot_params, trans_params], lr=0.05, betas=(0.9, 0.999))

    lss = []
    for i in range(iters):
        zoom_mat = torch.eye(4) * zoom_param
        transform_matrix = zoom_mat @ compute_rotation_matrix_from_ortho6d(rot_params.view(1,6))
        transform_matrix[0,:3,-1] += trans_params
        moving_centers_transformed = transform_matrix @ moving_centers.t()
        loss = torch.nn.functional.mse_loss(moving_centers_transformed[0].t(), fixed_centers)
        loss.backward()
        optim.step()
        optim.zero_grad()

    return transform_matrix.detach().clone().inverse()[0]



def get_centroid_reorient_grid_affine(moving_label, ref_filepath):
    DOF = 7
    assert ref_filepath.is_file(), f"ref_filepath {ref_filepath} is not a file"
    fixed_label = torch.as_tensor(nib.load(ref_filepath).get_fdata()).int()
    pt_heart_affine = register_centroids(fixed_label, moving_label, DOF)

    return pt_heart_affine