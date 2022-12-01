
import torch
import einops as eo
from slice_inflate.datasets.align_mmwhs import nifti_transform
from slice_inflate.utils.torch_utils import get_rotation_matrix_3d_from_angles

class AffineTransformModule(torch.nn.Module):
    def __init__(self, fov_mm, fov_vox, view_affine):
        super().__init__()

        self.fov_mm = fov_mm
        self.fov_vox = fov_vox
        self.fov_mm = fov_mm
        self.fov_vox = fov_vox
        self.view_affine = view_affine
        self.theta_t = torch.nn.Parameter(torch.zeros(1))
        self.theta_a = torch.nn.Parameter(torch.zeros(3))

    def get_batch_affine(self, batch_size):
        theta_m = angle_axis_to_rotation_matrix(self.theta_a.view(1,3))[0,:3,:3]
        theta_t = torch.cat([self.theta_t, torch.tensor([0,0], device=self.theta_t.device)])
        theta = torch.cat([theta_m, theta_t.view(3,1)], dim=1)
        theta = torch.cat([theta, torch.tensor(
            [0, 0, 0, 1], device=theta.device).view(1, 4)], dim=0)
        return theta.view(1, 4, 4).repeat(batch_size, 1, 1)

    def forward(self, x_image, x_label, nifti_affine, augment_affine, with_theta=True):

        x_image_is_none = x_image.numel() == 0 or x_image is None
        x_label_is_none = x_label.numel() == 0 or x_label is None

        assert not (x_image_is_none and x_label_is_none)

        device = x_label.device if not x_label_is_none else x_image.device
        B = x_label.shape[0] if not x_label_is_none else x_image.shape[0]
        new_affine = None

        # Compose the complete alignment from:
        # 1) view (sa, hla)
        # 2) learnt affine
        # 3) and align affine (global alignment @ augment, if any)

        if with_theta:
            theta = self.get_batch_affine(B)
        else:
            theta = torch.eye(4).view(1, 4, 4).repeat(B, 1, 1)

        final_affine = self.view_affine.to(device) @ augment_affine.to(device)

        # b = torch.load("/shared/slice_inflate/data/models/dulcet-salad-196_best/sa_atm.pth")
        # self.theta_t = torch.nn.Parameter(b['theta_t'])
        # self.theta_m = torch.nn.Parameter(b['theta_m'])
        # theta = self.get_batch_affine(B)
        # optimized_view_affine = theta.to(device) @ self.view_affine.to(device)
        # final_affine = optimized_view_affine @ augment_affine.to(device)
        # y_label, affine = nifti_transform(x_label, nifti_affine, final_affine,
        #                                     fov_mm=self.fov_mm, fov_vox=self.fov_vox, is_label=True)
        # nib.save(nib.Nifti1Image(y_label.int().detach().cpu().numpy()[0].argmax(0), affine=affine[0].cpu().detach().numpy()), "sa_initial.nii.gz")
        # nib.save(nib.Nifti1Image(y_label.int().detach().cpu().numpy()[0].argmax(0), affine=affine[0].cpu().detach().numpy()), "sa_learnt.nii.gz")

        if not x_image_is_none:
            y_image, new_affine = nifti_transform(x_image, nifti_affine, final_affine,
                                                  fov_mm=self.fov_mm, fov_vox=self.fov_vox, is_label=False,
                                                  pre_grid_sample_affine=theta)

        if not x_label_is_none:
            y_label, affine = nifti_transform(x_label, nifti_affine, final_affine,
                                              fov_mm=self.fov_mm, fov_vox=self.fov_vox, is_label=True,
                                              pre_grid_sample_affine=theta)

        return y_image, y_label, new_affine


class SoftCutModule(torch.nn.Module):
    def __init__(self, n_rows, n_cols, soft_cut_softness:float=8.0):
        super().__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.soft_cut_softness = soft_cut_softness
        self.offsets = torch.nn.Parameter(torch.zeros(n_rows, n_cols))

    def forward(self, b_volume):
        B,C,D,H,W = b_volume.shape
        # b_volume = eo.rearrange(b_volume, 'B C D H W -> D H B C W')
        b_volume = eo.rearrange(b_volume, 'B C D H W -> W B C D H')

        centers = (W-1)/2 + self.offsets * W/2

        sz_D = b_volume.shape[-2] // self.n_rows
        sz_H = b_volume.shape[-1] // self.n_cols

        n_dist = torch.distributions.normal.Normal(
            centers,
            torch.tensor(self.soft_cut_softness * W/2, device=centers.device))

        probs = (torch.arange(0, W).view(-1,1,1).repeat(1,self.n_rows,self.n_cols)).to(centers.device)
        probs = n_dist.log_prob(probs).exp()
        probs = probs / probs.max()
        probs = probs.to(b_volume.device)
        probs = probs.repeat_interleave(sz_D, dim=1).repeat_interleave(sz_H, dim=2)
        probs = probs.view(W,1,1,D,H)

        b_volume = (b_volume * probs).sum(0, keepdim=True)

        # Save b_volume as a nifti
        # import nibabel as nib
        # import numpy as np
        # nib.save(nib.Nifti1Image(b_volume[0,0][1:].sum(0,keepdim=True).detach().cpu().numpy(), affine=np.eye(4)), "b_volume.nii.gz")

        return eo.rearrange(b_volume, ' W B C D H -> B C D H W')

    def get_extra_state(self):
        state = dict(
            n_rows=self.n_rows,
            n_cols=self.n_cols
        )
        return state

    def set_extra_state(self, state):
        self.n_rows = state['n_rows']
        self.n_cols = state['n_cols']



def get_random_affine(angle_std, seed=0):
    angles = get_random_angles(angle_std, seed)
    theta_m = get_rotation_matrix_3d_from_angles(angles)

    theta = torch.eye(4)
    theta[:3,:3] = theta_m
    return theta



def get_random_angles(angle_std, seed=0):
    torch.random.manual_seed(seed)
    mean, std = torch.zeros(3), torch.ones(3) * angle_std
    angles = torch.normal(mean, std)
    return angles



def angle_axis_to_rotation_matrix(angle_axis):
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = tgm.angle_axis_to_rotation_matrix(input)  # Nx4x4
    """
    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        EPS = 1e-6
        theta = torch.sqrt(theta2+EPS)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4



def rotation_matrix_to_angle_axis(rotation_matrix):
    """Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    # todo add check that matrix is a valid rotation matrix
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)



def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (1 - mask_d0_d1.int())
    mask_c2 = (1 - mask_d2.int()) * mask_d0_nd1.int()
    mask_c3 = (1 - mask_d2.int()) * (1 - mask_d0_nd1.int())
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q



def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis



# based on:
# https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py#L138