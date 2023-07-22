from lib.losses3D.basic import *
from torch.nn import AdaptiveMaxPool3d
# Code was adapted and mofified from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py


class OCT2DBinaryCrossEntropyWithLogitsLoss(torch.nn.Module):
    """
    """

    def __init__(self, ignore_index=-1, path_size=32, bs=1):
        super(OCT2DBinaryCrossEntropyWithLogitsLoss, self).__init__()
        self.path_size = path_size
        self.bs = bs
        self.max_pool = AdaptiveMaxPool3d((1,path_size,path_size))
        
    def forward(self, input, target):
        if type(input) is tuple:
            if len(input) == 2:
                # DENSEVOXELNET
                loss1 = torch.nn.functional.binary_cross_entropy_with_logits(
                    self.max_pool(input[0].reshape((self.bs, -1, self.path_size, self.path_size))),
                    target
                )
                loss2 = torch.nn.functional.binary_cross_entropy_with_logits(
                    self.max_pool(input[1].reshape((self.bs, -1, self.path_size, self.path_size))),
                    target
                )
                loss = loss1 + 0.33*loss2
            elif len(input) == 4:
                # RESNET3DVAE
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    self.max_pool(input[0].reshape((self.bs, -1, self.path_size, self.path_size))),
                    target
                )
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                self.max_pool(input.reshape((self.bs, -1, self.path_size, self.path_size))),
                target
            )
        return loss, loss.clone().cpu().detach().numpy().reshape((1))