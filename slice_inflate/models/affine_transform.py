
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import numpy as np
import einops as eo
from slice_inflate.datasets.align_mmwhs import nifti_transform
from slice_inflate.utils.torch_utils import get_rotation_matrix_3d_from_angles

import dill
from slice_inflate.models.nnunet_models import Generic_UNet_Hybrid
from slice_inflate.utils.common_utils import get_script_dir
from pathlib import Path

class ConvNet(torch.nn.Module):
    def __init__(self, input_channels, kernel_size, inner_padding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(input_channels,32,kernel_size,padding=1), nn.BatchNorm3d(32), nn.LeakyReLU(),
            nn.AvgPool3d(2),
            nn.Conv3d(32,64,kernel_size,padding=inner_padding), nn.BatchNorm3d(64), nn.LeakyReLU(),
            nn.Conv3d(64,64,kernel_size,padding=inner_padding), nn.BatchNorm3d(64), nn.LeakyReLU(),
            nn.AvgPool3d(2),
            nn.Conv3d(64,64,kernel_size,padding=inner_padding), nn.BatchNorm3d(64), nn.LeakyReLU(),
            nn.Conv3d(64,64,kernel_size,padding=inner_padding), nn.BatchNorm3d(64), nn.LeakyReLU(),
            nn.AvgPool3d(2),
            nn.Conv3d(64,64,kernel_size,padding=inner_padding), nn.BatchNorm3d(64), nn.LeakyReLU(),
            nn.Conv3d(64,32,kernel_size,padding=inner_padding), nn.BatchNorm3d(32), nn.LeakyReLU(),
            nn.AvgPool3d(2),
            nn.Conv3d(32,32,kernel_size,padding=1), nn.BatchNorm3d(32), nn.LeakyReLU(),
            nn.Conv3d(32,1,1,padding=1), nn.BatchNorm3d(1)
        )


    def forward(self, x, encoder_only=False):
        return self.net(x)



class LocalisationNet(torch.nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        if True:
            self.conv_net = ConvNet(input_channels=input_channels, kernel_size=5, inner_padding=2)
            self.fc_in_num = 1*7**3
        else:
            init_dict_path = Path(get_script_dir(), "./slice_inflate/models/nnunet_init_dict_128_128_128.pkl")
            with open(init_dict_path, 'rb') as f:
                init_dict = dill.load(f)
            init_dict['num_classes'] = input_channels
            init_dict['deep_supervision'] = False
            init_dict['final_nonlin'] = torch.nn.Identity()
            use_skip_connections = False
            init_dict['norm_op'] = nn.BatchNorm3d
            init_dict['norm_op_kwargs'] = None
            nnunet_model = Generic_UNet(**init_dict, use_skip_connections=use_skip_connections, use_onehot_input=True)
            self.conv_net = nnunet_model
            self.fc_in_num = 256*16**3

        self.fca = nn.Linear(self.fc_in_num, 3)
        self.fct = nn.Linear(self.fc_in_num, 3)

    def forward(self, x):
        bsz = x.shape[0]
        h = self.conv_net(x, encoder_only=True)
        h = h.reshape(bsz, -1)
        theta_ap = self.fca(h)
        theta_tp = self.fct(h)
        return theta_ap, theta_tp
        # return theta_ap.atan(), 2.0*theta_tp.sigmoid()-1.0



class AffineTransformModule(torch.nn.Module):
    def __init__(self, input_channels,
        fov_mm, fov_vox, view_affine,
        init_theta_ap=None, init_theta_tp=None,
        optim_method='angle-axis', use_affine_theta=True, tag=None):

        super().__init__()
        assert optim_method in ['angle-axis', 'normal-vector'], \
            f"optim_method must be 'angle-axis' or 'normal-vector', not {optim_method}"
        self.optim_method = optim_method

        self.fov_mm = fov_mm
        self.fov_vox = fov_vox
        self.view_affine = view_affine.view(1,4,4)
        self.localisation_net = LocalisationNet(input_channels)

        self.use_affine_theta = use_affine_theta

        self.init_theta_ap = torch.nn.Parameter(torch.zeros(3), requires_grad=False)
        self.init_theta_tp = torch.nn.Parameter(torch.zeros(3), requires_grad=False)

        self.last_theta = None
        self.last_theta_a = None
        self.last_theta_t = None

        self.tag = tag

    def set_init_theta_ap(self, init_theta_ap):
        self.init_theta_ap.data = init_theta_ap.data

    def set_init_theta_tp(self, init_theta_tp):
        self.init_theta_tp.data = init_theta_tp.data

    def get_init_affines(self):
        assert not self.init_theta_ap.requires_grad and not self.init_theta_tp.requires_grad
        device = self.init_theta_tp.device
        theta_a = angle_axis_to_rotation_matrix(self.init_theta_ap.view(1,3))[0].view(1,4,4)
        theta_t = torch.cat([self.init_theta_tp, torch.tensor([1], device=device)])
        theta_t = torch.cat([torch.eye(4)[:4,:3].to(device), theta_t.view(4,1)], dim=1).view(1,4,4)

        assert theta_a.shape == theta_t.shape == (1,4,4)
        return theta_a.to(torch.float32), theta_t.to(torch.float32)

    def get_batch_affines(self, x):
        batch_size = x.shape[0]

        theta_ap, theta_tp = self.localisation_net(x.float())
        device = theta_ap.device

        # Apply initial rotation
        theta_ap = theta_ap + self.init_theta_ap.view(1,3).to(device)
        theta_tp = theta_tp + self.init_theta_tp.view(1,3).to(device)

        if self.optim_method == 'angle-axis':
            theta_ap[:,0] = 0.0 # [:,0] rotates in plane
        theta_tp[:,1:] = 0.0 # [:,0] is perpendicular to cut plane -> predict only this

        # theta_ap[:,1] = 0.0 # TODO remove that
        # theta_tp[...] = 0.0 # TODO remove that

        if self.optim_method == 'angle-axis':
            theta_a = angle_axis_to_rotation_matrix(theta_ap.view(batch_size,3))

        elif self.optim_method == 'normal-vector':
            theta_ap = theta_ap/theta_ap.norm(dim=1).view(-1,1) # Normalize
            theta_a = normal_to_rotation_matrix(theta_ap.view(batch_size,3))
        else:
            raise ValueError()

        theta_t = torch.cat([theta_tp, torch.ones(batch_size, device=device).view(batch_size,1)], dim=1)
        theta_t = torch.cat([
            torch.eye(4, device=device)[:4,:3].view(1,4,3).repeat(batch_size,1,1),
            theta_t.view(batch_size,4,1)
        ], dim=-1)

        assert theta_a.shape == theta_t.shape == (batch_size, 4, 4)
        return theta_a, theta_t

    def forward(self, x_image, x_label, nifti_affine, augment_affine, theta_override=None):

        x_image_is_none = x_image is None or x_image.numel() == 0
        x_label_is_none = x_label is None or x_label.numel() == 0

        assert not (x_image_is_none and x_label_is_none)

        y_image = None
        y_label = None
        device = x_label.device if not x_label_is_none else x_image.device
        B = x_label.shape[0] if not x_label_is_none else x_image.shape[0]

        # theta_ai = self.get_init_affine().to(device=device)

        if theta_override is not None:
            theta = theta_override.detach() # Caution: this is a non-differentiable theta
        else:
            if self.use_affine_theta:
                theta_a, theta_t = self.get_batch_affines(x_image) # Initial parameters are applied here as well
            else:
                theta_a, theta_t = self.get_init_affines()
                theta_a, theta_t = theta_a.to(device), theta_t.to(device)

            theta = theta_a @ theta_t

            self.last_theta = theta
            self.last_theta_a = theta_a
            self.last_theta_t = theta_t

        # Gef affines:
        # theta_a : Affine for learnt rotation and initialization of affine module
        # theta_t : Affine for learnt translation (shifts volume relative to the grid_sample D,H,W axes)
        # globabl_prelocate_affine : Affine for prelocating the volume (slice orientation and augmentation)
        # theta   : Affine for the learnt transformation

        global_prelocate_affine = self.view_affine.to(device) @ augment_affine.to(device)

        if not x_image_is_none:
            # nifti_affine is the affine of the original volume
            y_image, resampled_affine = nifti_transform(x_image, nifti_affine, global_prelocate_affine,
                                        fov_mm=self.fov_mm, fov_vox=self.fov_vox, is_label=False,
                                        pre_grid_sample_affine=theta)

        if not x_label_is_none:
            # nifti_affine is the affine of the original volume
            y_label, resampled_affine = nifti_transform(x_label, nifti_affine, global_prelocate_affine,
                                        fov_mm=self.fov_mm, fov_vox=self.fov_vox, is_label=True,
                                        pre_grid_sample_affine=theta)

        self.last_resampled_affine = resampled_affine

        return y_image, y_label, resampled_affine



class HardCutModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, b_volume):
        B,C,D,H,W = b_volume.shape
        b_volume = eo.rearrange(b_volume, 'B C D H W -> W B C D H')

        center = W//2
        cut = b_volume[center:center+1, ...]
        return eo.rearrange(cut, ' W B C D H -> B C D H W')



class SoftCutModule(torch.nn.Module):

    def __init__(self, soft_cut_softness:float=0.125):
        super().__init__()
        self.soft_cut_softness = soft_cut_softness

    def get_probs(self, W):
        center = (W-1)/2

        n_dist = torch.distributions.normal.Normal(
            center,
            torch.tensor(self.soft_cut_softness * W/2))

        probs = torch.arange(0, W)
        probs = n_dist.log_prob(probs).exp()
        probs = probs / probs.max()
        return probs

    def forward(self, b_volume):
        B,C,D,H,W = b_volume.shape
        b_volume = eo.rearrange(b_volume, 'B C D H W -> W B C D H')

        probs = self.get_probs(W).view(W,1,1,1,1).to(b_volume.device)
        b_volume = (b_volume * probs).sum(0, keepdim=True)

        return eo.rearrange(b_volume, ' W B C D H -> B C D H W')



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



def normal_to_rotation_matrix(normals):
    """Convert 3d vector (unnormalized normals) to 4x4 rotation matrix

    Args:
        normal (Tensor): tensor of 3d vector of normals.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = tgm.angle_axis_to_rotation_matrix(input)  # Nx4x4
    """

    nzs, nys, nxs = normals[:,0], normals[:,1], normals[:,2]

    # see https://math.stackexchange.com/questions/1956699/getting-a-transformation-matrix-from-a-normal-vector
    r00 = nys / torch.sqrt(nxs**2 + nys**2)
    r01 = -nxs / torch.sqrt(nxs**2 + nys**2)
    r02 = torch.zeros_like(nxs)
    r10 = nxs * nzs / torch.sqrt(nxs**2 + nys**2)
    r11 = nys * nzs / torch.sqrt(nxs**2 + nys**2)
    r12 = -torch.sqrt(nxs**2 + nys**2)
    r20 = nxs
    r21 = nys
    r22 = nzs
    zer = torch.zeros_like(nxs)
    one = torch.ones_like(nxs)

    theta_r = torch.stack(
        [r00, r01, r02, zer,
         r10, r11, r12, zer,
         r20, r21, r22, zer,
         zer, zer, zer, one], dim=1)

    theta_r = theta_r.view(-1,4,4)

    return theta_r



def angle_axis_to_rotation_matrix(angle_axis, eps=1e-6):
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
        theta = torch.sqrt(theta2+eps)
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
    with amp.autocast(enabled=False):
        _angle_axis = _angle_axis.to(torch.float32)
        theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2, eps)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
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



def rotation_matrix_to_angle_axis(rotation_matrix, eps=1e-6):
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
    return quaternion_to_angle_axis(quaternion, eps)



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
    with amp.autocast(enabled=False):
        t0_rep = t0_rep.to(torch.float32)
        mask_c0 = mask_c0.to(torch.float32)
        t1_rep = t1_rep.to(torch.float32)
        mask_c1 = mask_c1.to(torch.float32)
        t2_rep = t2_rep.to(torch.float32)
        mask_c2 = mask_c2.to(torch.float32)
        t3_rep = t3_rep.to(torch.float32)
        mask_c3 = mask_c3.to(torch.float32)

        q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                        t2_rep * mask_c2 + t3_rep * mask_c3 + eps)  # noqa
    q *= 0.5
    return q



def quaternion_to_angle_axis(quaternion: torch.Tensor, eps=1e-6) -> torch.Tensor:
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

    with amp.autocast(enabled=False):
        sin_squared_theta = sin_squared_theta.to(torch.float32)
        sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta+eps)

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



def get_theta_params(b_theta):
    bsz = b_theta.shape[0]
    theta_ap, theta_tp = rotation_matrix_to_angle_axis(b_theta[:,:3]), b_theta[:,:3,-1].view(bsz,3)
    return theta_ap, theta_tp



def get_mean_theta(b_theta, as_batch_size=False):
    bsz = b_theta.shape[0]
    theta_a, theta_t = get_theta_params(b_theta)
    mean_theta_a, mean_theta_t = theta_a.mean(0, keepdim=True), theta_t.mean(0, keepdim=True)
    mean_theta = angle_axis_to_rotation_matrix(mean_theta_a)
    mean_theta[:,:3,-1] = mean_theta_t

    if as_batch_size:
        mean_theta = mean_theta.view(1,4,4).repeat(bsz,1,1)

    return mean_theta