
import torch
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
        theta_m = get_rotation_matrix_3d_from_angles(self.theta_a)
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