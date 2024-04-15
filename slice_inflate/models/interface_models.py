import torch



class NNUNET_InterfaceModel(torch.nn.Module):
    def __init__(self, nnunet_model):
        super().__init__()
        self.nnunet_model = nnunet_model

    def forward(self, *args, **kwargs):
        y_hat = self.nnunet_model(*args, **kwargs)
        if isinstance(y_hat, tuple):
            return y_hat[0]
        else:
            return y_hat



class EPix2Vox_InterfaceModel(torch.nn.Module):
    def __init__(self, epix_model):
        super().__init__()
        self.epix_model = epix_model

    def forward(self, *args, **kwargs):
        input, epx = args
        slice_fg = [slc[:,1:].sum(dim=1, keepdim=True) for slc in input.chunk(2, dim=1)]
        slices = torch.cat(slice_fg, dim=1)
        slices = torch.nn.functional.interpolate(slices, size=[224,224]) # TODO automate
        B,N_SLICES,SPAT_H,SPAT_W = slices.shape
        slices = slices.view(B,N_SLICES,1,SPAT_H,SPAT_W).repeat(1,1,3,1,1) * 255.
        y_hat = self.epix_model(slices, epx)
        y_hat = y_hat.view(B,1,128,128,128) # TODO automate
        y_hat = torch.cat([1.-y_hat, y_hat], dim=1) # Create bg / fg channels
        return y_hat
