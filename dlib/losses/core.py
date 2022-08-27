import sys
from os.path import dirname, abspath

import re
import torch.nn as nn
import torch
import torch.nn.functional as F

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.losses.elb import ELB
from dlib.losses.entropy import Entropy
from dlib.crf.dense_crf_loss import DenseCRFLoss

__all__ = [
    'MasterLoss',
    'ClLoss',
    'ImgReconstruction',
    'SelfLearningFcams',
    'ConRanFieldFcams',
    'EntropyFcams',
    'MaxSizePositiveFcams'
]


class _ElementaryLoss(nn.Module):
    def __init__(self,
                 device=torch.device("cpu"),
                 name=None,
                 lambda_=1.,
                 elb=nn.Identity(),
                 logit=False,
                 support_background=False,
                 multi_label_flag=False,
                 sigma_rgb=15.,
                 sigma_xy=100.,
                 scale_factor=0.5,
                 start_epoch=None,
                 end_epoch=None,
                 seg_ignore_idx=-255
                 ):
        super(_ElementaryLoss, self).__init__()
        self._name = name
        self.lambda_ = lambda_
        self.elb = elb
        self.logit = logit
        self.support_background = support_background

        assert not multi_label_flag
        self.multi_label_flag = multi_label_flag

        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor

        if end_epoch == -1:
            end_epoch = None

        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.c_epoch = 0

        if self.logit:
            assert isinstance(self.elb, ELB)

        self.loss = None
        self._device = device

        self._zero = torch.tensor([0.0], device=self._device,
                                  requires_grad=False, dtype=torch.float)

        self.seg_ignore_idx = seg_ignore_idx

    def is_on(self, _epoch=None):
        if _epoch is None:
            c_epoch = self.c_epoch
        else:
            assert isinstance(_epoch, int)
            c_epoch = _epoch

        if (self.start_epoch is None) and (self.end_epoch is None):
            return True

        l = [c_epoch, self.start_epoch, self.end_epoch]
        if all([isinstance(z, int) for z in l]):
            return self.start_epoch <= c_epoch <= self.end_epoch

        if self.start_epoch is None and isinstance(self.end_epoch, int):
            return c_epoch <= self.end_epoch

        if isinstance(self.start_epoch, int) and self.end_epoch is None:
            return c_epoch >= self.start_epoch

        return False

    def unpacke_low_cams(self, cams_low, glabel):
        n = cams_low.shape[0]
        select_lcams = [None for _ in range(n)]

        for i in range(n):
            llabels = [glabel[i]]

            if self.support_background:
                llabels = [xx + 1 for xx in llabels]
                llabels = [0] + llabels

            for l in llabels:
                tmp = cams_low[i, l, :, :].unsqueeze(
                        0).unsqueeze(0)
                if select_lcams[i] is None:
                    select_lcams[i] = tmp
                else:
                    select_lcams[i] = torch.cat((select_lcams[i], tmp), dim=1)

        return select_lcams

    def update_t(self):
        if isinstance(self.elb, ELB):
            self.elb.update_t()

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            out = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
            if isinstance(self.elb, ELB):
                out = out + '_elb'
            if self.logit:
                out = out + '_logit'
            return out
        else:
            return self._name

    def forward(self,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None
                ):
        self.c_epoch = epoch


class ClLoss(_ElementaryLoss):
    def __init__(self, **kwargs):
        super(ClLoss, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(reduction="mean").to(self._device)

    def forward(self,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None
                ):
        super(ClLoss, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        return self.loss(input=cl_logits, target=glabel) * self.lambda_


class ImgReconstruction(_ElementaryLoss):
    def __init__(self, **kwargs):
        super(ImgReconstruction, self).__init__(**kwargs)

        self.loss = nn.MSELoss(reduction="none").to(self._device)

    def forward(self,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None
                ):
        super(ImgReconstruction, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        n = x_in.shape[0]
        loss = self.elb(self.loss(x_in, im_recon).view(n, -1).mean(
            dim=1).view(-1, ))
        return self.lambda_ * loss.mean()


class SelfLearningFcams(_ElementaryLoss):
    def __init__(self, **kwargs):
        super(SelfLearningFcams, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.seg_ignore_idx).to(self._device)

    def forward(self,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None
                ):
        super(SelfLearningFcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag

        return self.loss(input=fcams, target=seeds) * self.lambda_


class ConRanFieldFcams(_ElementaryLoss):
    def __init__(self, **kwargs):
        super(ConRanFieldFcams, self).__init__(**kwargs)

        self.loss = DenseCRFLoss(
            weight=self.lambda_, sigma_rgb=self.sigma_rgb,
            sigma_xy=self.sigma_xy, scale_factor=self.scale_factor
        ).to(self._device)

    def forward(self,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None
                ):
        super(ConRanFieldFcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero
        
        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        return self.loss(images=raw_img, segmentations=fcams_n)


class EntropyFcams(_ElementaryLoss):
    def __init__(self, **kwargs):
        super(EntropyFcams, self).__init__(**kwargs)

        self.loss = Entropy().to(self._device)

    def forward(self,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None
                ):
        super(EntropyFcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert fcams.ndim == 4
        bsz, c, h, w = fcams.shape

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        probs = fcams_n.permute(0, 2, 3, 1).contiguous().view(bsz * h * w, c)

        return self.lambda_ * self.loss(probs).mean()


class MaxSizePositiveFcams(_ElementaryLoss):
    def __init__(self, **kwargs):
        super(MaxSizePositiveFcams, self).__init__(**kwargs)

        assert isinstance(self.elb, ELB)

    def forward(self,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None
                ):
        super(MaxSizePositiveFcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag  # todo

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        n = fcams.shape[0]
        loss = None
        for c in [0, 1]:
            bl = fcams_n[:, c].view(n, -1).sum(dim=-1).view(-1, )
            if loss is None:
                loss = self.elb(-bl)
            else:
                loss = loss + self.elb(-bl)

        return self.lambda_ * loss * (1./2.)


class MasterLoss(nn.Module):
    def __init__(self, device=torch.device("cpu"), name=None):
        super().__init__()
        self._name = name

        self.losses = []
        self.l_holder = []
        self.n_holder = [self.__name__]
        self._device = device

    def add(self, loss_: _ElementaryLoss):
        self.losses.append(loss_)
        self.n_holder.append(loss_.__name__)

    def update_t(self):
        for loss in self.losses:
            loss.update_t()

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name

    def forward(self, **kwargs):
        assert self.losses != []

        self.l_holder = []
        for loss in self.losses:
            self.l_holder.append(loss(**kwargs).to(self._device))

        loss = sum(self.l_holder)
        self.l_holder = [loss] + self.l_holder
        return loss

    def to_device(self):
        for loss in self.losses:
            loss.to(self._device)

    def check_losses_status(self):
        print('-' * 60)
        print('Losses status:')

        for i, loss in enumerate(self.losses):
            if hasattr(loss, 'is_on'):
                print(self.n_holder[i+1], ': ... ',
                      loss.is_on(),
                      "({}, {})".format(loss.start_epoch, loss.end_epoch))
        print('-' * 60)

    def __str__(self):
        return "{}():".format(
            self.__class__.__name__, ", ".join(self.n_holder))


if __name__ == "__main__":
    from dlib.utils.reproducibility import set_seed
    set_seed(seed=0)
    b, c = 10, 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss = MasterLoss(device=device)
    print(loss.__name__, loss, loss.l_holder, loss.n_holder)
    loss.add(SelfLearningFcams())
    for l in loss.losses:
        print(l, isinstance(l, SelfLearningFcams))

    for e in loss.n_holder:
        print(e)

