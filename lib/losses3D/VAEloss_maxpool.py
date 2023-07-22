from lib.losses3D.basic import *
from torch.nn import AdaptiveMaxPool3d
# Code was adapted and mofified from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py



class OCT2DBinaryCrossEntropyWithLogitsLossVAE(torch.nn.Module):
    """
    """

    def __init__(self, ignore_index=-1, path_size=32, bs=1, type="BCE", h1=0.1,h2=0.1):
        super(OCT2DBinaryCrossEntropyWithLogitsLossVAE, self).__init__()
        self.path_size = path_size
        self.bs = bs
        self.max_pool = AdaptiveMaxPool3d((1,path_size,path_size))
        self.type = type
        self.h1 = h1
        self.h2 = h2
        
    def forward(self, input, target):
        # RESNET3DVAE
        recon_x, vae_out, mu, logvar = input
        recon_x = self.max_pool(recon_x.reshape((self.bs, -1, self.path_size, self.path_size)))
        recon_x = torch.nn.functional.softmax(recon_x)
        rec_flat = recon_x.view(self.bs, -1)
        x_flat = target.view(self.bs, -1)

        if self.type=="BCE":
            loss_rec = torch.nn.functional.binary_cross_entropy(rec_flat,x_flat, reduction='sum')
        elif type=="L1":
            loss_rec = torch.sum(torch.abs(rec_flat-x_flat))
        elif type =="L2":
            loss_rec = torch.sum(torch.sqrt(rec_flat*rec_flat - x_flat*x_flat))

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = loss_rec*self.h1 + KLD*self.h2

        return loss, loss.clone().cpu().detach().numpy().reshape((1))