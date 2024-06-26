import torch



class EPix2Vox_InterfaceModel(torch.nn.Module):
    def __init__(self, epix_model):
        super().__init__()
        self.epix_model = epix_model

    def forward(self, *args, **kwargs):
        input, epx = args
        slice_fg = [slc[:,1:].sum(dim=1, keepdim=True) for slc in input.chunk(2, dim=1)]
        slices = torch.cat(slice_fg, dim=1)
        slices = torch.nn.functional.interpolate(slices, size=[224,224])
        B,N_SLICES,SPAT_H,SPAT_W = slices.shape
        slices = slices.view(B,N_SLICES,1,SPAT_H,SPAT_W).repeat(1,1,3,1,1) * 255.
        y_hat = self.epix_model(slices, epx)
        out_sz = self.epix_model.out_size
        y_hat = y_hat.view(B,1,out_sz,out_sz,out_sz)
        y_hat = torch.cat([1.-y_hat, y_hat], dim=1) # Create bg / fg channels
        return y_hat
