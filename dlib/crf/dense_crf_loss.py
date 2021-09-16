import sys
import os
from os.path import dirname, abspath, join
import datetime as dt

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

sys.path.append(
    join(root_dir,
         "crf/crfwrapper/bilateralfilter/build/lib.linux-x86_64-3.7")
)

from bilateralfilter import bilateralfilter, bilateralfilter_batch


__all__ = ['DenseCRFLoss']


class DenseCRFLossFunction(Function):
    
    @staticmethod
    def forward(ctx,
                images,
                segmentations,
                sigma_rgb,
                sigma_xy
                ):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape
        
        # ROIs = ROIs.unsqueeze_(1).repeat(1, ctx.K, 1, 1)
        # segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        # ctx.ROIs = ROIs
        
        densecrf_loss = 0.0
        images = images.numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H,
                              ctx.W, sigma_rgb, sigma_xy)
        densecrf_loss -= np.dot(segmentations, AS)
    
        # averaged by the number of images
        densecrf_loss /= ctx.N
        
        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = - 2 * grad_output * torch.from_numpy(ctx.AS)/ctx.N
        grad_segmentation = grad_segmentation.cuda()
        # grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None


class DenseCRFLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        """
        Init. function.
        :param weight: float. It is Lambda for the crf loss.
        :param sigma_rgb: float. sigma for the bilateheral filtering (
        appearance kernel): color similarity.
        :param sigma_xy: float. sigma for the bilateral filtering
        (appearance kernel): proximity.
        :param scale_factor: float. ratio to scale the image and
        segmentation. Helpful to control the computation (speed) / precision.
        """
        super(DenseCRFLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor
    
    def forward(self, images, segmentations):
        """
        Forward loss.
        Image and segmentation are scaled with the same factor.

        :param images: torch tensor of the image (values in [0, 255]). shape
        N*C*H*W. DEVICE: CPU.
        :param segmentations: softmaxed logits.
        :return: loss score (scalar).
        """
        scaled_images = F.interpolate(images,
                                      scale_factor=self.scale_factor,
                                      mode='nearest',
                                      recompute_scale_factor=False
                                      )
        scaled_segs = F.interpolate(segmentations,
                                    scale_factor=self.scale_factor,
                                    mode='bilinear',
                                    recompute_scale_factor=False,
                                    align_corners=False)

        val = self.weight * DenseCRFLossFunction.apply(
            scaled_images,
            scaled_segs,
            self.sigma_rgb,
            self.sigma_xy * self.scale_factor
        )
        return val

    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )


def test_DenseCRFLoss():
    from dlib.utils.reproducibility import set_seed
    from dlib.utils.shared import announce_msg

    seed = 0
    cuda = "1"
    print("cuda:{}".format(cuda))
    print("DEVICE BEFORE: ", torch.cuda.current_device())
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    set_seed(seed=seed)
    n, h, w = 2, 500, 500
    scale_factor = 0.5
    img = torch.randint(
        low=0, high=256,
        size=(n, 3, h, w), dtype=torch.float, device=DEVICE,
        requires_grad=False).cpu()
    nbr_cl = 20
    segmentations = torch.rand(size=(n, nbr_cl, h, w), dtype=torch.float,
                               device=DEVICE,
                               requires_grad=True)
    loss = DenseCRFLoss(weight=1e-7,
                        sigma_rgb=15.,
                        sigma_xy=100.,
                        scale_factor=scale_factor
                        ).to(DEVICE)
    announce_msg("testing {}".format(loss))
    set_seed(seed=seed)
    if nbr_cl > 1:
        softmax = nn.Softmax(dim=1)
    else:
        softmax = nn.Sigmoid()

    t0 = dt.datetime.now()
    z = loss(images=img, segmentations=softmax(segmentations))
    print('Loss: {} (nbr_cl: {})'.format(z, nbr_cl))
    print('Time ({} x {} : scale: {}: N: {}): {}'.format(
        h, w, scale_factor, n, dt.datetime.now() - t0))


if __name__ == '__main__':
    test_DenseCRFLoss()
