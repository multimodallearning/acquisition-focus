import torch
from torch.utils.checkpoint import checkpoint
from dynamic_network_architectures.architectures.unet import PlainConvUNet

from slice_inflate.utils.nifti_utils import get_zooms, rescale_rot_components_with_diag
from slice_inflate.utils.torch_utils import set_module


class HybridUnet(PlainConvUNet):
    def __init__(self, n_slices, num_classes):

        n_stages = 6
        features_per_stage=[n_slices*c for c in [16,32,64,128,256,256]]

        super().__init__(
            input_channels=n_slices*num_classes,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=torch.nn.modules.conv.Conv3d,
            kernel_sizes=n_stages*[[3,3,3]],
            strides=[[1,1,1]]+(n_stages-1)*[[2,2,2]],
            n_conv_per_stage=n_stages*[2],
            num_classes=num_classes,
            n_conv_per_stage_decoder=(n_stages-1)*[2],
            conv_bias=True,
            norm_op=torch.nn.modules.instancenorm.InstanceNorm3d,
            norm_op_kwargs=dict(eps=1e-05, affine=True),
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=torch.nn.modules.activation.LeakyReLU,
            nonlin_kwargs=dict(inplace=True),
            deep_supervision=False,
            nonlin_first=False
        )
        
        self.skip_connector = SkipConnector(n_slices)
        self.__setup_2d_encoder__(n_slices)
        self.n_slices = n_slices

    def forward(self, x, b_grid_affines):
        skips = self.encoder(x)
        embedded_skips = [self.skip_connector(s, b_grid_affines) for s in skips]
        return self.decoder(embedded_skips)

    def __setup_2d_encoder__(self, n_slices):
        for keychain, mod in self.encoder.named_modules(remove_duplicate=False):
            if isinstance(mod, torch.nn.modules.conv.Conv3d):
                replacer = torch.nn.Conv2d(mod.in_channels, mod.out_channels,
                                           kernel_size=mod.kernel_size[:2],
                                           stride=mod.stride[:2], padding=mod.padding[:2],
                                           groups=n_slices)

            elif isinstance(mod, torch.nn.modules.instancenorm.InstanceNorm3d):
                replacer = torch.nn.InstanceNorm2d(mod.num_features,
                                                   mod.eps, mod.momentum, mod.affine,
                                                   mod.track_running_stats)
            else:
                continue

            set_module(self.encoder, keychain, replacer)




class SkipConnector(torch.nn.Module):
    def __init__(self, n_slices):
        self.dtype = torch.float32
        self.n_slices = n_slices
        super().__init__()

    def forward(self, x, b_grid_affines):
        B,C,SPAT,_ = x.shape
        C_PER_SLICE = C//self.n_slices
        target_shape = torch.Size([B,C_PER_SLICE,SPAT,SPAT,SPAT])
        zer = torch.zeros(B,C,SPAT,SPAT,SPAT).to(x)
        zer[..., SPAT//2] = x # Embed slice in center of last dimension
        x = zer

        x_sa, x_hla = torch.chunk(x, self.n_slices, dim=1)

        # Grid sample first channel chunk with inverse SA affines
        rescaled_sa_affines = b_grid_affines[0]
        # Rescale to sample from volume slice space into volume space (forward grid sampling sampled to single slice, non-stacked)
        rescaled_sa_affines = rescale_rot_components_with_diag(rescaled_sa_affines, 1/get_zooms(rescaled_sa_affines))

        sa_grid = torch.nn.functional.affine_grid(
            rescaled_sa_affines.to(self.dtype).inverse()[:,:3,:].view(B,3,4), target_shape, align_corners=False
        )
        transformed_sa = checkpoint(torch.nn.functional.grid_sample,
            x_sa, sa_grid.to(x_sa), 'bilinear', 'zeros', False
        )

        # Grid sample second channel chunk with inverse HLA affines
        rescaled_hla_affines = b_grid_affines[1]
        # Rescale to sample from volume slice space into volume space (forward grid sampling sampled to single slice, non-stacked)
        rescaled_hla_affines = rescale_rot_components_with_diag(rescaled_hla_affines, 1/get_zooms(rescaled_hla_affines))

        hla_grid = torch.nn.functional.affine_grid(
            rescaled_hla_affines.to(self.dtype).inverse()[:,:3,:].view(B,3,4), target_shape, align_corners=False,
        )
        transformed_hla = checkpoint(torch.nn.functional.grid_sample,
            x_hla, hla_grid.to(x_hla), 'bilinear', 'zeros', False,
        )

        skip_out = torch.cat([transformed_sa, transformed_hla], dim=1)
        return skip_out