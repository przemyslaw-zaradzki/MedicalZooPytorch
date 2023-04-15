from lib.losses3D.basic import *
from torch.nn import AdaptiveMaxPool3d
# Code was adapted and mofified from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py


class OCT2DBinaryCrossEntropyWithLogitsLossNoMaxPool(torch.nn.Module):
    """
    """

    def __init__(self, ignore_index=-1):
        super(OCT2DBinaryCrossEntropyWithLogitsLossNoMaxPool, self).__init__()

    def forward(self, input, target):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input[0], 
            target
        )
        return loss, loss.clone().cpu().detach().numpy().reshape((1))