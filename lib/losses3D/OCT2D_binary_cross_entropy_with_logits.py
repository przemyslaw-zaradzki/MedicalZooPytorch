from lib.losses3D.basic import *
from torch.nn import AdaptiveMaxPool3d
# Code was adapted and mofified from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py


class OCT2DBinaryCrossEntropyWithLogitsLoss(torch.nn.Module):
    """
    """

    def __init__(self, ignore_index=-1):
        super(OCT2DBinaryCrossEntropyWithLogitsLoss, self).__init__()
        self.max_pool = AdaptiveMaxPool3d((1,32,32))

    def forward(self, input, target):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            self.max_pool(input.reshape((1,-1,32,32))), 
            target
        )
        return loss, loss.clone().cpu().detach().numpy().reshape((1))