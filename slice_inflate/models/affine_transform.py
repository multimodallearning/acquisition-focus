
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import numpy as np
import einops as eo
from slice_inflate.utils.nifti_utils import nifti_grid_sample
from slice_inflate.utils.torch_utils import determine_network_output_size

import dill
from slice_inflate.models.nnunet_models import Generic_UNet_Hybrid
from slice_inflate.utils.common_utils import get_script_dir
from slice_inflate.utils.torch_utils import torch_manual_seeded
from pathlib import Path

class ConvNet(torch.nn.Module):
    def __init__(self, input_channels, kernel_size, padding, norm_op=nn.InstanceNorm3d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size, padding=padding), norm_op(32), nn.LeakyReLU(),
            nn.AvgPool3d(2),
            nn.Conv3d(32, 64, kernel_size, padding=padding), norm_op(64), nn.LeakyReLU(),
            nn.Conv3d(64, 64, kernel_size, padding=padding), norm_op(64), nn.LeakyReLU(),
            nn.AvgPool3d(2),
            nn.Conv3d(64, 64, kernel_size, padding=padding), norm_op(64), nn.LeakyReLU(),
            nn.Conv3d(64, 64, kernel_size, padding=padding), norm_op(64), nn.LeakyReLU(),
            nn.AvgPool3d(2),
            nn.Conv3d(64, 64, kernel_size, padding=padding), norm_op(64), nn.LeakyReLU(),
            nn.Conv3d(64, 32, kernel_size, padding=padding), norm_op(32), nn.LeakyReLU(),
            nn.AvgPool3d(2),
            nn.Conv3d(32, 32, kernel_size, padding=padding), norm_op(32), nn.LeakyReLU(),
            nn.Conv3d(32, 1, 1, padding=0), norm_op(1)
        )


    def forward(self, x, encoder_only=False):
        return self.net(x)



class LocalisationNet(torch.nn.Module):
    def __init__(self, input_channels, output_size, size_3d):
        super().__init__()

        self.conv_net = ConvNet(
            input_channels=input_channels,
            kernel_size=5,
            padding=2,
            # norm_op=nn.BatchNorm3d
        )
        sz = determine_network_output_size(
            self.conv_net,
            torch.zeros([1,input_channels, *size_3d])
        )
        self.fc_in_num = torch.tensor(sz).prod().int().item()
        self.fc = nn.Linear(self.fc_in_num, output_size)

    def forward(self, x):
        bsz = x.shape[0]
        h = self.conv_net(x, encoder_only=True)
        h = h.reshape(bsz, -1)
        h = self.fc(h)
        return h



class AffineTransformModule(torch.nn.Module):
    def __init__(self, input_channels, size_3d,
        fov_mm, fov_vox, view_affine,
        optim_method='angle-axis', use_affine_theta=True,
        offset_clip_value=1., zoom_clip_value=2., tag=None,
        align_corners=False):

        super().__init__()
        assert fov_vox[0] == fov_vox[1] == fov_vox[2]
        assert optim_method in ['angle-axis', 'normal-vector', 'R6-vector'], \
            f"optim_method must be 'angle-axis', 'normal-vector' or 'R6-vector', not {optim_method}"

        self.optim_method = optim_method

        if optim_method == 'angle-axis':
            self.ap_space = 3
            self.optim_function = angle_axis_to_rotation_matrix
            self.init_theta_ap = torch.nn.Parameter(torch.zeros(self.ap_space), requires_grad=False)

        elif optim_method == 'normal-vector':
            self.ap_space = 3
            self.optim_function = normal_to_rotation_matrix
            self.init_theta_ap = torch.nn.Parameter(torch.zeros(self.ap_space), requires_grad=False)

        elif optim_method == 'R6-vector':
            self.ap_space = 6
            self.optim_function = compute_rotation_matrix_from_ortho6d
            self.init_theta_ap = torch.nn.Parameter(torch.tensor([[1e-2,0,0,0,1e-2,0]]), requires_grad=False)

        else:
            raise ValueError()

        self.fov_mm = fov_mm
        self.fov_vox = fov_vox
        self.spat = int(fov_vox[-1])

        if isinstance(view_affine, torch.Tensor):
            self.view_affine = view_affine.view(1,4,4)
        else:
            self.view_affine = None

        self.use_affine_theta = use_affine_theta
        self.align_corners = align_corners

        self.offset_clip_value = offset_clip_value
        self.zoom_clip_value = zoom_clip_value

        self.vox_range = int(round(
            self.get_vox_offsets_from_gs_offsets(self.offset_clip_value)
            - self.get_vox_offsets_from_gs_offsets(-self.offset_clip_value)
        ))

        self.arra = torch.arange(0, self.vox_range) + (self.spat - self.vox_range) // 2

        self.localisation_net = LocalisationNet(
            input_channels,
            self.ap_space + 3*int(self.vox_range) + 1, # 3*spatial dimension for translational parameters and 1x zoom parameter
            size_3d=size_3d
        )

        self.use_affine_theta = use_affine_theta
        self.tag = tag
        self.align_corners = align_corners

        self.init_theta_t_offsets = torch.nn.Parameter(torch.zeros([3]), requires_grad=False)
        self.init_theta_zp = torch.nn.Parameter(torch.ones([1,1]), requires_grad=False)

        self.last_theta_ap = None
        self.last_theta_t_offsets = None
        self.last_theta_zp = None
        self.last_theta_a = None
        self.last_theta_t = None
        self.last_theta = None
        self.last_grid_affine = None
        self.last_transformed_nii_affine = None

    def set_init_theta_ap(self, init_theta_ap):
        self.init_theta_ap.data = init_theta_ap.data

    def set_init_theta_t_offsets(self, init_theta_t_offsets):
        self.init_theta_t_offsets.data = init_theta_t_offsets.data

    def set_init_theta_zp(self, init_theta_zp):
        self.init_theta_zp.data = init_theta_zp.data


    def get_init_affines(self):
        assert not self.init_theta_ap.requires_grad \
            and not self.init_theta_t_offsets.requires_grad \
            and not self.init_theta_zp.requires_grad

        device = self.init_theta_t_offsets.device
        theta_a = self.optim_function(self.init_theta_ap.view(1,self.ap_space))[0].view(1,4,4)
        theta_t = torch.cat([self.init_theta_t_offsets, torch.tensor([1], device=device)])
        theta_t = torch.cat([torch.eye(4)[:4,:3].to(device), theta_t.view(4,1)], dim=1).view(1,4,4)
        theta_z = torch.diag_embed(torch.cat([
            self.init_theta_zp,
            self.init_theta_zp,
            self.init_theta_zp,
            torch.ones([1,1], device=device)], dim=-1
        ))

        assert theta_a.shape == theta_t.shape == theta_z.shape == (1,4,4)
        return theta_a.to(torch.float32), theta_t.to(torch.float32), theta_z.to(torch.float32)

    def get_gs_offsets_from_theta_tp(self, theta_tp):
        assert theta_tp.shape[1:] == (3,self.vox_range)
        slice_posxs = (
            torch.nn.functional.softmax(theta_tp, dim=2)
            * self.arra.to(theta_tp).view(1,1,self.vox_range)
        ).sum(-1)

        if self.align_corners:
            # see https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/GridSampler.h
            gs_offsets = 2./(self.spat-1)*slice_posxs - 1.0
        else:
            gs_offsets = (2.0*slice_posxs + 1.0)/self.spat - 1.0

        return gs_offsets

    def get_vox_offsets_from_gs_offsets(self, gs_offsets):

        if self.align_corners:
            # see https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/GridSampler.h
            vox_offsets = (gs_offsets + 1.0) * (self.spat-1) / 2.0
        else:
            vox_offsets = ((gs_offsets+1.0)*self.spat - 1.0) / 2.0

        return vox_offsets

    def get_batch_affines(self, x):
        B,C,D,H,W = x.shape

        theta_atz_p = self.localisation_net(x.float())
        device = theta_atz_p.device
        theta_ap = theta_atz_p[:,:self.ap_space]
        theta_tp = theta_atz_p[:,self.ap_space:-1].view(B,3,self.vox_range)
        theta_zp = theta_atz_p[:,-1:]

        # Apply initial values
        theta_ap = theta_ap + self.init_theta_ap.view(1,self.ap_space).to(device)
        # TODO: What about translational values?
        theta_zp = theta_zp + self.init_theta_zp.view(1,1).to(device)

        # Rotation matrix definition
        theta_ap = theta_ap.view(B, self.ap_space)

        if self.optim_method == 'angle-axis':
            theta_ap[:,0] = 0.0 # [:,0] rotates in plane TODO this should not be restricted
        elif self.optim_method == 'normal-vector':
            theta_ap = theta_ap/theta_ap.norm(dim=1).view(-1,1) # Normalize

        theta_a = self.optim_function(theta_ap)

        # Translation matrix definition
        theta_t_offsets = self.get_gs_offsets_from_theta_tp(theta_tp)
        # theta_t_offsets[:, 1:] = 0. # TODO check
        theta_t = torch.cat([theta_t_offsets, torch.ones(B, device=device).view(B,1)], dim=1)
        theta_t = torch.cat([
            torch.eye(4, device=device)[:4,:3].view(1,4,3).repeat(B,1,1),
            theta_t.view(B,4,1)
        ], dim=-1)

        # Zoom matrix definition
        theta_zp = self.zoom_clip_value * -(theta_zp.tanh()) + 1. # Zoom needs to be in range(1.0 +- clip value)
        theta_z = torch.diag_embed(torch.cat([
            theta_zp,
            theta_zp,
            theta_zp,
            torch.ones([B,1], device=device)], dim=-1)
        )

        assert theta_a.shape == theta_t.shape == theta_z.shape == (B, 4, 4)

        self.last_theta_ap = theta_ap
        self.last_theta_t_offsets = theta_t_offsets
        self.last_theta_zp = theta_zp
        self.last_theta_a = theta_a
        self.last_theta_t = theta_t
        self.last_theta_z = theta_z

        return theta_a, theta_t, theta_z

    def forward(self, x_image, x_label, nifti_affine, known_augment_affine, hidden_augment_affine, theta_override=None):

        x_image_is_none = x_image is None or x_image.numel() == 0
        x_label_is_none = x_label is None or x_label.numel() == 0

        assert not (x_image_is_none and x_label_is_none)

        y_image = None
        y_label = None
        device = x_label.device if not x_label_is_none else x_image.device
        B = x_label.shape[0] if not x_label_is_none else x_image.shape[0]

        if theta_override is not None:
            theta = theta_override.detach().clone() # Caution: this is a non-differentiable theta
        else:
            theta_a, theta_t, theta_z = self.get_init_affines()
            theta_a, theta_t, theta_z = theta_a.to(device), theta_t.to(device), theta_z.to(device)
            theta_a, theta_t, theta_z = theta_a.repeat(B,1,1), theta_t.repeat(B,1,1), theta_z.repeat(B,1,1)

            if self.use_affine_theta:
                theta_a_b, theta_t_b, theta_z_b = self.get_batch_affines(x_image) # Initial parameters are applied here as well
                theta_a = theta_a_b @ theta_a
                theta_t = theta_t_b @ theta_t
                theta_z = theta_z_b @ theta_z
            else:
                self.last_theta_ap = None
                self.last_theta_t_offsets = None
                self.last_theta_zp = None
                self.last_theta_a = None
                self.last_theta_t = None
                self.last_theta_z = None

            theta = theta_z @ theta_a @ theta_t

            self.last_theta = theta

        # Gef affines:
        # theta_a : Affine for learnt rotation and initialization of affine module
        # theta_t : Affine for learnt translation (shifts volume relative to the grid_sample D,H,W axes)
        # theta_z : Affine for learnt zoom
        # globabl_prelocate_affine : Affine for prelocating the volume (slice orientation and augmentation)
        # theta   : Affine for the learnt transformation

        if self.view_affine is not None:
            global_prelocate_affine = self.view_affine.repeat(B,1,1).to(device)
        else:
            global_prelocate_affine = None

        if not x_image_is_none:
            # nifti_affine is the affine of the original volume
            y_image, grid_affine, transformed_nii_affine = nifti_grid_sample(x_image, nifti_affine, global_prelocate_affine,
                                        fov_mm=self.fov_mm, fov_vox=self.fov_vox, is_label=False,
                                        pre_grid_sample_affine=known_augment_affine.to(theta.device) @ theta,
                                        pre_grid_sample_hidden_affine=hidden_augment_affine)

        if not x_label_is_none:
            # nifti_affine is the affine of the original volume
            y_label, grid_affine, transformed_nii_affine = nifti_grid_sample(x_label, nifti_affine, global_prelocate_affine,
                                        fov_mm=self.fov_mm, fov_vox=self.fov_vox, is_label=True,
                                        pre_grid_sample_affine=known_augment_affine.to(theta.device) @ theta,
                                        pre_grid_sample_hidden_affine=hidden_augment_affine)

        self.last_grid_affine = grid_affine
        self.last_transformed_nii_affine = transformed_nii_affine

        return y_image, y_label, grid_affine, transformed_nii_affine



class HardCutModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, b_volume):
        B,C,D,H,W = b_volume.shape
        b_volume = eo.rearrange(b_volume, 'B C D H W -> W B C D H')

        center = W//2
        cut = b_volume[center:center+1, ...]
        return eo.rearrange(cut, ' W B C D H -> B C D H W')



# class LearnCutModule(torch.nn.Module):

#     def __init__(self, input_channels, size_3d, align_corners=False, mode='linear'):
#         super().__init__()
#         assert mode in ['linear', 'nearest']
#         assert size_3d[0] == size_3d[1] == size_3d[2]
#         self.localisation_net = LocalisationNet(
#             input_channels,
#             size_3d[-1],
#             size_3d=size_3d[-1]*3) # 3 for three translational parameters
#         self.size_3d = size_3d
#         self.arra = torch.arange(0, size_3d[-1])
#         self.align_corners = align_corners
#         self.mode = mode

    # def forward(self, b_volume, slice_posxs_override=None):
    #     assert b_volume.dim() == 5
    #     B,C,D,H,W = b_volume.shape
    #     assert D == H == W
    #     assert W == self.size_3d[-1]

    #     if slice_pos_override is None:
    #         b_slice_selector = self.localisation_net(b_volume).view(B,3,W)
    #         slice_posxs = (
    #             torch.nn.functional.softmax(b_slice_selector, dim=2)
    #             * self.arra.to(b_volume).view(1,1,W)
    #         ).sum(-1)
    #     else:
    #         slice_posxs = slice_posxs_override

    #     slice_posxs[:,0] = slice_posxs[:,0].clamp(0,W-1)
    #     slice_posxs[:,1] = slice_posxs[:,1].clamp(0,W-1)
    #     slice_posxs[:,2] = slice_posxs[:,2].clamp(0,W-1)

    #     slice_posxs = torch.tensor([0, 0, 80.]).to(b_volume) # TODO remove!

    #     gs_space_affine = torch.eye(4).repeat(B,1,1).to(b_volume)

    #     if self.align_corners:
    #         # see https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/GridSampler.h
    #         gs_offsets = 2./(W-1)*slice_posxs - 1.0
    #     else:
    #         gs_offsets = (2.0*slice_posxs + 1.0)/W - 1.0

    #     gs_space_affine[:,:3,-1] = gs_offsets # Caution: W is at row index 0

    #     grid = torch.nn.functional.affine_grid(
    #         gs_space_affine[:,:3,:].view(B,3,4), (B,C,D,H,1), align_corners=self.align_corners
    #     ).to(device=volume.device)

    #     cut_slice = checkpoint(
    #         torch.nn.functional.grid_sample,
    #         volume.to(dtype=dtype), grid.to(dtype=dtype), self.mode, 'zeros', align_corners=self.align_corners
    #     )

    #     # slice_lower_posxs = slice_posxs.floor().clamp(0,W-1)
    #     # slice_upper_posxs = (slice_lower_posxs+1).clamp(0,W-1)
    #     # upper_factor = slice_posxs - slice_lower_posxs
    #     # lower_factor = 1.0 - upper_factor

    #     # if self.mode == 'nearest':
    #     #     if lower_factor >= upper_factor:
    #     #         slice_upper_posxs = slice_lower_posxs
    #     #     else:
    #     #         slice_lower_posxs = slice_upper_posxs
    #     #     lower_factor = upper_factor = 0.5

    #     # # Gradient flows through slice interpolation factors
    #     # slice_lower_posxs = slice_lower_posxs.long()
    #     # slice_upper_posxs = slice_upper_posxs.long()

    #     # interpolated_slice = (
    #     #     lower_factor * b_volume[..., slice_lower_posxs[2]]
    #     #     + upper_factor * b_volume[..., slice_upper_posxs[2]]
    #     # )

    #     return cut_slice, gs_space_affine


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

def get_random_ortho6_vector(rotation_strength=0.2, constrained=True):
    params = torch.tensor([[1.,0.,0., 0.,1.,0.]])
    rand_r = torch.rand_like(params) * rotation_strength - rotation_strength/2
    if constrained:
        return params + rand_r
    return rand_r

def get_random_affine(rotation_strength=0.2, zoom_strength=0.2):
    rand_z = torch.rand(1) * zoom_strength - zoom_strength/2 + 1.0
    # rand_theta_r = compute_rotation_matrix_from_ortho6d(get_random_ortho6_vector(rotation_strength))

    ortho_vect = torch.tensor((rotation_strength*torch.randn(2)).tolist()+[1.])
    ortho_vect /= ortho_vect.norm(2)
    one = torch.tensor([1.]+(rotation_strength*torch.randn(2)).tolist())
    two = torch.cross(ortho_vect, one)
    two /= two.norm(2)
    one = torch.cross(two, ortho_vect)

    rand_theta_r = torch.stack([one,two,ortho_vect])
    rand_theta_z = torch.diag(torch.tensor([rand_z,rand_z,rand_z,1.0]))

    return rand_theta_z @ rand_theta_r



def compute_rotation_matrix_from_ortho6d(ortho):
    # see https://github.com/papagina/RotationContinuity/blob/master/Inverse_Kinematics/code/tools.py
    x_raw = ortho[:, 0:3]
    y_raw = ortho[:, 3:6]

    x = x_raw / x_raw.norm(dim=1, keepdim=True)
    z = x.cross(y_raw)
    z = z / z.norm(dim=1, keepdim=True)
    y = z.cross(x)

    # torch.stack([x, y, z], dim=-1)
    r00 = x[:,0]
    r01 = y[:,0]
    r02 = z[:,0]
    r10 = x[:,1]
    r11 = y[:,1]
    r12 = z[:,1]
    r20 = x[:,2]
    r21 = y[:,2]
    r22 = z[:,2]
    zer = torch.zeros_like(r00)
    one = torch.ones_like(r00)

    theta_r = torch.stack(
        [r00, r01, r02, zer,
         r10, r11, r12, zer,
         r20, r21, r22, zer,
         zer, zer, zer, one], dim=1)

    theta_r = theta_r.view(-1,4,4)

    return theta_r



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
    B = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(B, 1, 1)
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
    raise NotImplementedError("Zooms parameters missing.")
    bsz = b_theta.shape[0]
    theta_ap, theta_tp = rotation_matrix_to_angle_axis(b_theta[:,:3]), b_theta[:,:3,-1].view(bsz,3)
    return theta_ap, theta_tp



def get_mean_theta(b_theta, as_B=False):
    bsz = b_theta.shape[0]
    theta_a, theta_t = get_theta_params(b_theta)
    mean_theta_a, mean_theta_t = theta_a.mean(0, keepdim=True), theta_t.mean(0, keepdim=True)
    mean_theta = angle_axis_to_rotation_matrix(mean_theta_a)
    mean_theta[:,:3,-1] = mean_theta_t

    if as_B:
        mean_theta = mean_theta.view(1,4,4).repeat(bsz,1,1)

    return mean_theta