from pathlib import Path

import torch
import torch.nn as nn

from slice_inflate.utils.nifti_utils import nifti_grid_sample
from slice_inflate.utils.torch_utils import determine_network_output_size
from slice_inflate.utils.transform_utils import angle_axis_to_rotation_matrix, get_random_affine, compute_rotation_matrix_from_ortho6d, normal_to_rotation_matrix
from slice_inflate.functional.clinical_cardiac_views import get_inertia_tensor, get_main_principal_axes, get_pix_affine_from_center_and_plane_vects, get_torch_grid_affine_from_pix_affine



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



class LocalizationNet(torch.nn.Module):
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
    def __init__(self, input_channels,
        volume_fov_mm, volume_fov_vox,
        slice_fov_mm, slice_fov_vox,
        optim_method='angle-axis', use_affine_theta=True,
        offset_clip_value=1., zoom_clip_value=2., view_id=None,
        align_corners=False, rotate_slice_to_min_principle=False):

        super().__init__()
        assert volume_fov_mm[0] == volume_fov_mm[1] == volume_fov_mm[2]
        assert volume_fov_vox[0] == volume_fov_vox[1] == volume_fov_vox[2]
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

        self.volume_fov_mm = volume_fov_mm
        self.volume_fov_vox = volume_fov_vox
        self.slice_fov_vox = slice_fov_vox
        self.slice_fov_mm = slice_fov_mm

        self.use_affine_theta = use_affine_theta
        self.align_corners = align_corners
        self.rotate_slice_to_min_principle = rotate_slice_to_min_principle

        self.offset_clip_value = offset_clip_value
        self.zoom_clip_value = zoom_clip_value

        self.spat = volume_fov_vox[0]

        self.vox_range = (
            self.get_vox_offsets_from_gs_offsets(self.offset_clip_value)
            - self.get_vox_offsets_from_gs_offsets(-self.offset_clip_value)
        ).round().int().item()
        self.arra = torch.arange(0, self.vox_range) + (self.spat - self.vox_range) // 2

        self.localization_net = LocalizationNet(
            input_channels,
            self.ap_space + 3*self.vox_range + 1, # 3*spatial dimension for translational parameters and 1x zoom parameter
            size_3d=volume_fov_vox
        )

        self.view_id = view_id
        self.align_corners = align_corners

        self.init_theta_t_offsets = torch.nn.Parameter(torch.zeros([3]), requires_grad=False)
        self.init_theta_zp = torch.nn.Parameter(torch.ones([1,1]), requires_grad=False)

        self.last_theta = None
        self.last_grid_affine = None
        self.last_transformed_nifti_affine = None
        self.random_grid_affine = get_random_affine(rotation_strength=4., zoom_strength=0.)[None]

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

        theta_atz_p = self.localization_net(x)
        device = theta_atz_p.device
        theta_ap = theta_atz_p[:,:self.ap_space]
        theta_tp = theta_atz_p[:,self.ap_space:-1].view(B,3,self.vox_range)
        theta_zp = theta_atz_p[:,-1:]

        # Apply initial values
        theta_ap = theta_ap + self.init_theta_ap.view(1,self.ap_space).to(device)
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
        if self.offset_clip_value == 0.:
            theta_t_offsets = 0. * theta_t_offsets
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

        return theta_a, theta_t, theta_z

    def forward(self, x_soft_label, x_label, x_image, nifti_affine, grid_affine_pre_mlp,
                # grid_affine_augment,
                theta_override=None):

        x_soft_label_is_none = x_soft_label is None or x_soft_label.numel() == 0
        x_label_is_none = x_label is None or x_label.numel() == 0
        x_image_is_none = x_image is None or x_image.numel() == 0

        assert not x_soft_label_is_none

        y_soft_label = None
        y_label = None
        y_image = None
        device = x_soft_label.device
        B = x_soft_label.shape[0]

        with torch.no_grad():
            # This is the affine that is applied to the volume before the grid sampling -> Input of MLP must be reoriented to a common space
            if not x_soft_label_is_none:
                # nifti_affine is the affine of the original volume
                x_soft_label_pre_mlp, _, transformed_nii_affine = nifti_grid_sample(x_soft_label, nifti_affine,
                    target_fov_mm=self.volume_fov_mm, target_fov_vox=self.volume_fov_vox, is_label=False,
                    pre_grid_sample_affine=grid_affine_pre_mlp
                )
                # import nibabel as nib
                # nib.Nifti1Image(x_soft_label_pre_mlp[0].argmax(0).detach().cpu().int().numpy(), affine=transformed_nii_affine[0].detach().cpu().numpy()).to_filename('x_soft_label_pre_mlp.nii.gz')

        if theta_override is not None:
            theta = theta_override.detach().clone() # Caution: this is a non-differentiable theta
        else:
            theta_a, theta_t, theta_z = self.get_init_affines()
            theta_a, theta_t, theta_z = theta_a.to(device), theta_t.to(device), theta_z.to(device)
            theta_a, theta_t, theta_z = theta_a.repeat(B,1,1), theta_t.repeat(B,1,1), theta_z.repeat(B,1,1)

            if self.use_affine_theta:
                theta_a_b, theta_t_b, theta_z_b = self.get_batch_affines(x_soft_label_pre_mlp) # Initial parameters are applied here as well
                theta_a = theta_a @ theta_a_b
                theta_t = theta_t @ theta_t_b
                theta_z = theta_z @ theta_z_b

            theta = theta_t @ theta_a @ theta_z

            self.last_theta = theta

        # Gef affines:
        # theta_a : Affine for learnt rotation and initialization of affine module
        # theta_t : Affine for learnt translation (shifts volume relative to the grid_sample D,H,W axes)
        # theta_z : Affine for learnt zoom
        # global_prelocate_affine : Affine for prelocating the volume (slice orientation and augmentation)
        # theta   : Affine for the learnt transformation

        # Here is the learnable grid_sampling
        grid_affine_pre_mlp = grid_affine_pre_mlp.to(theta)
        if not x_soft_label_is_none:
            # nifti_affine is the affine of the original volume
            y_soft_label, grid_affine, transformed_nii_affine = nifti_grid_sample(x_soft_label, nifti_affine,
                target_fov_mm=self.slice_fov_mm, target_fov_vox=self.slice_fov_vox, is_label=False,
                pre_grid_sample_affine=grid_affine_pre_mlp @ theta,
            )

        with torch.no_grad():
            if not x_label_is_none:
                # nifti_affine is the affine of the original volume
                y_label, _, _ = nifti_grid_sample(x_label, nifti_affine,
                    target_fov_mm=self.slice_fov_mm, target_fov_vox=self.slice_fov_vox, is_label=True,
                    pre_grid_sample_affine=grid_affine_pre_mlp @ theta,
                )

            if not x_image_is_none:
                # nifti_affine is the affine of the original volume
                y_image, _, _ = nifti_grid_sample(x_image, nifti_affine,
                    target_fov_mm=self.slice_fov_mm, target_fov_vox=self.slice_fov_vox, is_label=False,
                    pre_grid_sample_affine=grid_affine_pre_mlp @ theta,
                    # pre_grid_sample_hidden_affine=grid_affine_augment
                )

        # Make sure the grid_affines only contain rotational components
        # assert torch.allclose(
        #     (grid_affine[:,:3,:3] @ grid_affine[:,:3,:3].transpose(-1,-2)),
        #     torch.eye(3)[None].repeat(B,1,1).to(grid_affine),
        #     atol=1e-4
        # )

        if self.rotate_slice_to_min_principle:
            # Rotate to main principle of slice to constrain the output
            y_soft_label, align_affine, transformed_nii_affine = rotate_slice_to_min_principle(y_soft_label,
                transformed_nii_affine, is_label=False)

            with torch.no_grad():
                if not x_label_is_none:
                    y_label, _, transformed_nii_affine = rotate_slice_to_min_principle(y_label,
                        transformed_nii_affine, is_label=True, align_affine_override=align_affine)
                if not x_image_is_none:
                    y_image, _, transformed_nii_affine = rotate_slice_to_min_principle(y_image,
                        transformed_nii_affine, is_label=False, align_affine_override=align_affine)

            grid_affine = grid_affine @ align_affine

        self.last_grid_affine = grid_affine
        self.last_transformed_nifti_affine = transformed_nii_affine

        return y_soft_label, y_label, y_image, grid_affine, transformed_nii_affine



def rotate_slice_to_min_principle(x_input, nii_affine, is_label=False, align_affine_override=None):
    assert x_input.shape[-1] == 1
    B = x_input.shape[0]

    if align_affine_override is None:
        b_align_affines = torch.zeros(B,4,4).to(x_input.device)

        with torch.no_grad():
            for b_idx, lbl in enumerate(x_input):
                center, I = get_inertia_tensor(lbl.argmax(0))
                center[-1] = 0.5
                min_principal, *_ = get_main_principal_axes(I)
                second_vect = min_principal.cross(torch.tensor([0.,0.,1.]))
                pix_align_affine = get_pix_affine_from_center_and_plane_vects(
                    center, min_principal, second_vect,
                    do_return_normal_three=False
                )
                pt_align_affine = get_torch_grid_affine_from_pix_affine(pix_align_affine, x_input.shape[-3:])
                b_align_affines[b_idx] = pt_align_affine

    else:
        b_align_affines = align_affine_override

    # Rotate according to min-principle direction
    # Flipping is done later but can be inherited here already, when align_affine_override is not None
    y_output, b_align_affines, transformed_nii_affine = nifti_grid_sample(x_input, nii_affine,
                                                                    pre_grid_sample_affine=b_align_affines,
                                                                    is_label=is_label)

    return y_output, b_align_affines, transformed_nii_affine



class ATModulesContainer(torch.nn.ModuleList):
    def __init__(self, config, num_classes):
        super().__init__(self)

        for view_id in config.view_ids:
            self.add_new_atm(view_id, config, num_classes)
        self.is_optimized = torch.nn.Parameter(torch.tensor(len(config.view_ids) * [False]), requires_grad=False)

    def add_new_atm(self, view_id, config, num_classes):
        atm = AffineTransformModule(num_classes,
            torch.tensor(config.prescan_fov_mm),
            torch.tensor(config.prescan_fov_vox),
            torch.tensor(config.slice_fov_mm),
            torch.tensor(config.slice_fov_vox),
            offset_clip_value=config.offset_clip_value,
            zoom_clip_value=config.zoom_clip_value,
            optim_method=config.affine_theta_optim_method,
            view_id=view_id,
            rotate_slice_to_min_principle=config.rotate_slice_to_min_principle)

        self.append(atm)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict.update({'is_optimized': self.get_active_views()})
        return state_dict

    def get_active_views(self):
        return self.is_optimized.cpu() | self.get_all_atms_requires_grad()

    def get_active_view_modules(self):
        return [mod for mod, is_active in zip(self,self.get_active_views()) if is_active]

    def get_all_atms_requires_grad(self):
        return torch.tensor([next(atm.localization_net.parameters()).requires_grad for atm in self])

    def get_next_non_optimized_view_module(self):
        next_idx = self.__get_next_non_optimized_idx__()
        if next_idx is not None:
            return self[next_idx]
        return

    def __get_next_non_optimized_idx__(self):
        if False in self.is_optimized:
            return (self.is_optimized == False).nonzero()[0]
        return