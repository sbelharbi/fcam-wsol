import sys
from os.path import dirname, abspath

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from kornia.morphology import dilation
from kornia.morphology import erosion
from skimage.filters import threshold_otsu
from skimage import filters

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


__all__ = ['GetPseudoMaskSLFCAMS',
           'GetAdaptivePseudoMaskSLFCAMS',
           'GetFastSeederSLFCAMS']


class GetFastSeederSLFCAMS(nn.Module):
    def __init__(self,
                 min_: int,
                 max_: int,
                 min_p: float,
                 fg_erode_k: int,
                 fg_erode_iter: int,
                 ksz: int,
                 device,
                 support_background: bool,
                 multi_label_flag: bool,
                 seg_ignore_idx: int
                 ):
        super(GetFastSeederSLFCAMS, self).__init__()
        assert not multi_label_flag

        self._device = device

        assert isinstance(ksz, int)
        assert ksz > 0
        self.ksz = ksz
        self.kernel = None
        if self.ksz > 1:
            self.kernel = torch.ones((self.ksz, self.ksz), dtype=torch.long,
                                     device=self._device)

        assert isinstance(min_, int)
        assert isinstance(max_, int)
        assert min_ >= 0
        assert max_ >= 0
        assert min_ + max_ > 0

        self.min_ = min_
        self.max_ = max_

        assert isinstance(min_p, float)
        assert 0. <= min_p <= 1.
        self.min_p = min_p

        # fg
        assert isinstance(fg_erode_k, int)
        assert fg_erode_k >= 1
        self.fg_erode_k = fg_erode_k

        # fg
        assert isinstance(fg_erode_iter, int)
        assert fg_erode_iter >= 0
        self.fg_erode_iter = fg_erode_iter

        self.fg_kernel_erode = None
        if self.fg_erode_iter > 0:
            assert self.fg_erode_k > 1
            self.fg_kernel_erode = torch.ones(
                (self.fg_erode_k, self.fg_erode_k), dtype=torch.long,
                device=self._device)

        self.support_background = support_background
        self.multi_label_flag = multi_label_flag

        self.ignore_idx = seg_ignore_idx

    def erosion_n(self, x):
        assert self.fg_erode_k > 1
        assert x.ndim == 4
        assert x.shape[0] == 1
        assert x.shape[1] == 1

        out: torch.Tensor = x
        tmp = x
        for i in range(self.fg_erode_iter):
            out = erosion(out, self.fg_kernel_erode)
            if out.sum() == 0:
                out = tmp
                break
            else:
                tmp = out

        assert 0 <= torch.min(out) <= 1
        assert 0 <= torch.max(out) <= 1
        assert out.shape == x.shape

        return out

    def dilate(self, x):
        assert self.ksz > 1
        assert x.ndim == 4
        assert x.shape[0] == 1
        assert x.shape[1] == 1

        out: torch.Tensor = dilation(x, self.kernel)
        assert 0 <= torch.min(out) <= 1
        assert 0 <= torch.max(out) <= 1
        assert out.shape == x.shape

        return out

    def identity(self, x):
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 4

        assert 0 <= torch.min(x) <= 1
        assert 0 <= torch.max(x) <= 1

        if self.ksz == 1:
            dilate = self.identity
        elif self.ksz > 1:
            dilate = self.dilate
        else:
            raise ValueError

        if self.fg_erode_iter > 0:
            erode = self.erosion_n
        else:
            erode = self.identity

        b, d, h, w = x.shape
        assert d == 1  # todo multilabl.
        out = torch.zeros((b, h, w), dtype=torch.long, requires_grad=False,
                          device=self._device)
        nbr_bg = int(self.min_p * h * w)

        for i in range(b):
            cam = x[i].squeeze()  # h, w
            cam_img = (cam.cpu().detach().numpy() * 255).astype(np.uint8)
            _bad_egg = False

            fg = torch.zeros((h, w), dtype=torch.long, device=self._device,
                             requires_grad=False)

            bg = torch.zeros((h, w), dtype=torch.long, device=self._device,
                             requires_grad=False)

            if cam_img.min() == cam_img.max():
                _bad_egg = True

            if not _bad_egg:
                otsu_thresh = threshold_otsu(cam_img)
                if otsu_thresh == 0:
                    otsu_thresh = 1
                if otsu_thresh == 255:
                    otsu_thresh = 254

                # li_thres = filters.threshold_li(cam_img,
                # initial_guess=otsu_thresh)
                ROI = torch.from_numpy(cam_img > otsu_thresh).to(self._device)
                ROI = erode(ROI.unsqueeze(0).unsqueeze(0) * 1).squeeze()

                # fg
                idx_fg = torch.nonzero(ROI, as_tuple=True)  # (idx, idy)
                n_fg = idx_fg[0].numel()
                if n_fg >= 0:
                    if self.max_ > 0:
                        probs = torch.ones(n_fg, dtype=torch.float)
                        selected = probs.multinomial(
                            num_samples=min(self.max_, n_fg), replacement=False)
                        fg[idx_fg[0][selected], idx_fg[1][selected]] = 1
                        fg = dilate(fg.view(1, 1, h, w)).squeeze()

                # bg
                val, idx_bg_ = torch.sort(cam.view(h * w), dim=0,
                                          descending=False)
                tmp = bg * 1.
                if nbr_bg > 0:
                    tmp = tmp.view(h * w)
                    tmp[idx_bg_[:nbr_bg]] = 1
                    tmp = tmp.view(h, w)

                    idx_bg = torch.nonzero(tmp, as_tuple=True)  #
                    # (idx, idy)
                    n_bg = idx_bg[0].numel()
                    if n_bg >= 0:
                        if self.min_ > 0:
                            probs = torch.ones(n_bg, dtype=torch.float)
                            selected = probs.multinomial(
                                num_samples=min(self.min_, n_bg),
                                replacement=False)
                            bg[idx_bg[0][selected], idx_bg[1][selected]] = 1
                            bg = dilate(bg.view(1, 1, h, w)).squeeze()

            # sanity
            outer = fg + bg
            fg[outer == 2] = 0
            bg[outer == 2] = 0

            seeds = torch.zeros((h, w), dtype=torch.long, device=self._device,
                                requires_grad=False) + self.ignore_idx

            seeds[fg == 1] = 1
            seeds[bg == 1] = 0

            out[i] = seeds.detach().clone()

        assert out.dtype == torch.long
        return out

    def extra_repr(self):
        return 'min_={}, max_={}, ' \
               'ksz={}, min_p: {}, fg_erode_k: {}, fg_erode_iter: {}, ' \
               'support_background={},' \
               'multi_label_flag={}, seg_ignore_idx={}'.format(
                self.min_, self.max_, self.ksz, self.min_p, self.fg_erode_k,
                self.fg_erode_iter,
                self.support_background,
                self.multi_label_flag, self.ignore_idx)


class GetAdaptivePseudoMaskSLFCAMS(nn.Module):
    def __init__(self,
                 min_: int,
                 max_: int,
                 ksz: int,
                 device,
                 support_background: bool,
                 multi_label_flag: bool,
                 seg_ignore_idx: int
                 ):
        super(GetAdaptivePseudoMaskSLFCAMS, self).__init__()
        assert not multi_label_flag

        self._device = device

        assert isinstance(ksz, int)
        assert ksz > 0
        self.ksz = ksz
        self.kernel = None
        if self.ksz > 1:
            self.kernel = torch.ones((self.ksz, self.ksz), dtype=torch.long,
                                     device=self._device)

        assert isinstance(min_, int)
        assert isinstance(max_, int)
        assert min_ >= 0
        assert max_ >= 0
        assert min_ + max_ > 0

        self.min_ = min_
        self.max_ = max_

        self.support_background = support_background
        self.multi_label_flag = multi_label_flag

        self.ignore_idx = seg_ignore_idx

    def dilate(self, x):
        assert self.ksz > 1
        assert x.ndim == 4
        assert x.shape[0] == 1
        assert x.shape[1] == 1

        out: torch.Tensor = dilation(x, self.kernel)
        assert 0 <= torch.min(out) <= 1
        assert 0 <= torch.max(out) <= 1
        assert out.shape == x.shape

        return out

    def identity(self, x):
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 4

        assert 0 <= torch.min(x) <= 1
        assert 0 <= torch.max(x) <= 1

        if self.ksz == 1:
            dilate = self.identity
        elif self.ksz > 1:
            dilate = self.dilate
        else:
            raise ValueError

        b, d, h, w = x.shape
        assert d == 1  # todo multilabl.
        out = torch.zeros((b, h, w), dtype=torch.long, requires_grad=False,
                          device=self._device)

        for i in range(b):
            cam = x[i].squeeze()  # h, w
            cam_img = (cam.cpu().detach().numpy() * 255).astype(np.uint8)
            otsu_thresh = threshold_otsu(cam_img)
            if otsu_thresh == 0:
                otsu_thresh = 1
            if otsu_thresh == 255:
                otsu_thresh = 254

            li_thres = filters.threshold_li(cam_img, initial_guess=otsu_thresh)
            ROI = torch.from_numpy(cam_img > li_thres).to(self._device)

            # fg
            fg = torch.zeros((h, w), dtype=torch.long, device=self._device,
                             requires_grad=False)
            idx_fg = torch.nonzero(ROI, as_tuple=True)  # (idx, idy)
            n_fg = idx_fg[0].numel()
            if n_fg >= 0:
                if self.max_ > 0:
                    probs = torch.ones(n_fg, dtype=torch.float)
                    selected = probs.multinomial(
                        num_samples=min(self.max_, n_fg), replacement=False)
                    fg[idx_fg[0][selected], idx_fg[1][selected]] = 1
                    fg = dilate(fg.view(1, 1, h, w)).squeeze()

            # bg
            bg = torch.zeros((h, w), dtype=torch.long, device=self._device,
                             requires_grad=False)
            idx_bg = torch.nonzero(torch.logical_not(ROI), as_tuple=True)  #
            # (idx, idy)
            n_bg = idx_bg[0].numel()
            if n_bg >= 0:
                if self.min_ > 0:
                    probs = torch.ones(n_bg, dtype=torch.float)
                    selected = probs.multinomial(
                        num_samples=min(self.min_, n_bg), replacement=False)
                    bg[idx_bg[0][selected], idx_bg[1][selected]] = 1
                    bg = dilate(bg.view(1, 1, h, w)).squeeze()

            # sanity

            outer = fg + bg
            fg[outer == 2] = 0
            bg[outer == 2] = 0

            seeds = torch.zeros((h, w), dtype=torch.long, device=self._device,
                                requires_grad=False) + self.ignore_idx

            seeds[fg == 1] = 1
            seeds[bg == 1] = 0

            out[i] = seeds.detach().clone()

        assert out.dtype == torch.long
        return out

    def extra_repr(self):
        return 'min_={}, max_={}, ' \
               'ksz={}, support_background={},' \
               'multi_label_flag={}, seg_ignore_idx={}'.format(
                self.min_, self.max_, self.ksz, self.support_background,
                self.multi_label_flag, self.ignore_idx)


class GetPseudoMaskSLFCAMS(nn.Module):
    def __init__(self,
                 min_: int,
                 max_: int,
                 min_ext: int,
                 max_ext: int,
                 block: int,
                 ksz: int,
                 device,
                 support_background: bool,
                 multi_label_flag: bool,
                 seg_ignore_idx: int
                 ):
        super(GetPseudoMaskSLFCAMS, self).__init__()

        assert not multi_label_flag  # todo

        self._device = device

        assert isinstance(ksz, int)
        assert ksz > 0
        self.ksz = ksz
        self.kernel = None
        if self.ksz > 1:
            self.kernel = torch.ones((self.ksz, self.ksz), dtype=torch.long,
                                     device=self._device)

        assert isinstance(min_, int)
        assert isinstance(max_, int)
        assert min_ >= 0
        assert max_ >= 0
        assert min_ + max_ > 0

        self.min_ = min_
        self.max_ = max_

        assert block > 0
        self.block = block

        assert isinstance(max_ext, int)
        assert isinstance(min_ext, int)
        assert min_ext >= min_
        assert max_ext >= max_
        self.min_ext = min_ext
        self.max_ext = max_ext

        self.support_background = support_background
        self.multi_label_flag = multi_label_flag

        self.ignore_idx = seg_ignore_idx

    def dilate(self, x):
        assert self.ksz > 1
        assert x.ndim == 4
        assert x.shape[0] == 1
        assert x.shape[1] == 1

        out: torch.Tensor = dilation(x, self.kernel)
        assert 0 <= torch.min(out) <= 1
        assert 0 <= torch.max(out) <= 1
        assert out.shape == x.shape

        return out

    def identity(self, x):
        return x

    def select_with_block(self, cam: torch.Tensor, up: bool) -> torch.Tensor:
        assert self.block > 1

        assert cam.ndim == 4
        assert cam.shape[0] == cam.shape[1] == 1

        h, w = cam.shape[2:]
        h_ = int(h / self.block) * self.block
        w_ = int(w / self.block) * self.block

        unfold = torch.nn.Unfold(kernel_size=(self.block, self.block),
                                 dilation=1, padding=0, stride=self.block)
        blocks = unfold(cam)  # 1, blockxblock, l
        blocks = blocks.transpose(1, 2)  # 1, l, blockxblock
        scores = blocks.squeeze(0).mean(dim=-1)  # l
        val_, idx_ = torch.sort(scores, dim=0, descending=False)

        subholder = blocks.detach().clone() * 0  # 1, l, blockxblock
        if up:
            idx__ = idx_[-self.max_ext:]
            nbl = self.max_
        else:
            idx__ = idx_[:self.min_ext]
            nbl = self.min_

        pblocks = torch.ones(idx__.numel(), dtype=torch.float)
        idxblocks = pblocks.multinomial(num_samples=nbl, replacement=False)

        sel = torch.ones(self.block * self.block, dtype=torch.float)

        for idxb in idxblocks:
            i = sel.multinomial(num_samples=1, replacement=False)
            subholder[0, idx__[idxb], i] = 1.

        subholder = subholder.transpose(1, 2)  # 1, blockxblock, l

        folder = torch.nn.Fold(output_size=(h_, w_),
                               kernel_size=(self.block, self.block),
                               dilation=1, padding=0, stride=self.block)

        subholder_ = folder(subholder)
        assert subholder_.shape == (1, 1, h_, w_)
        assert subholder_.dtype == torch.float

        roi = torch.zeros((h, w), dtype=torch.long, device=self._device,
                          requires_grad=False)
        roi[:h_, :w_] = subholder_

        return roi.view(h * w).long()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)

        if self.ksz == 1:
            dilate = self.identity
        elif self.ksz > 1:
            dilate = self.dilate
        else:
            raise ValueError

        b, d, h, w = x.shape
        assert d == 1  # todo multilabl.
        out = torch.zeros((b, h, w), dtype=torch.long, requires_grad=False,
                          device=self._device)

        for i in range(b):

            fg = torch.zeros((h, w), dtype=torch.long, device=self._device,
                             requires_grad=False)
            fg = fg.view(h * w)

            t = x[i].squeeze().view(h * w)

            val, idx = torch.sort(t, dim=0, descending=False)
            if self.max_ > 0:
                if self.block == 1:
                    if self.max_ == self.max_ext:
                        fg[idx[-self.max_:]] = 1  # foreground
                    else:
                        idx__ = idx[-self.max_ext:]
                        probs = torch.ones(idx__.numel(), dtype=torch.float)
                        new_idx = probs.multinomial(num_samples=self.max_,
                                                    replacement=False)
                        fg[idx__[new_idx]] = 1
                else:
                    fg = self.select_with_block(x[i].unsqueeze(0), up=True)
                fg = dilate(fg.view(1, 1, h, w)).contiguous().view(h * w)

            bg = fg.clone() * 0
            if self.min_ > 0:
                if self.block == 1:
                    if self.min_ == self.min_ext:
                        bg[idx[0:self.min_]] = 1  # background
                    else:
                        idx__ = idx[:self.min_ext]
                        probs = torch.ones(idx__.numel(), dtype=torch.float)
                        new_idx = probs.multinomial(num_samples=self.min_,
                                                    replacement=False)
                        bg[idx__[new_idx]] = 1
                else:
                    bg = self.select_with_block(x[i].unsqueeze(0), up=False)
                bg = dilate(bg.view(1, 1, h, w)).contiguous().view(h * w)

            outer = fg + bg
            fg[outer == 2] = 0
            bg[outer == 2] = 0

            seed = fg.clone() * 0 + self.ignore_idx
            seed[fg == 1] = 1
            seed[bg == 1] = 0

            out[i] = seed.view(h, w).detach().clone()

        assert out.dtype == torch.long
        return out

    def extra_repr(self):
        return 'min_={}, max_={}, min_ext={}, max_ext={}, ' \
               'block={}, ksz={}, support_background={},' \
               'multi_label_flag={}, seg_ignore_idx={}'.format(
                self.min_, self.max_, self.min_ext, self.max_ext, self.block,
                self.ksz, self.support_background, self.multi_label_flag,
                self.ignore_idx)


def test_GetFastSeederSLFCAMS():
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap

    def get_cm():
        col_dict = dict()
        for i in range(256):
            col_dict[i] = 'k'

        col_dict[0] = 'k'
        col_dict[int(255 / 2)] = 'b'
        col_dict[255] = 'r'
        colormap = ListedColormap([col_dict[x] for x in col_dict.keys()])

        return colormap

    def cam_2Img(_cam):
        return _cam.squeeze().cpu().numpy().astype(np.uint8) * 255

    def plot_limgs(_lims):
        nrows = 1
        ncols = len(_lims)

        him, wim = _lims[0][0].shape
        r = him / float(wim)
        fw = 10
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))
        for i, (im, tag) in enumerate(_lims):
            axes[0, i].imshow(im, cmap=get_cm())
            axes[0, i].text(3, 40, tag,
                            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8})

        plt.show()

    cuda = "1"
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    seed = 0
    min_ = 10
    max_ = 10
    min_p = .2
    fg_erode_k = 11
    fg_erode_iter = 1

    cam = torch.rand((1, 1, 224, 224), dtype=torch.float,
                     device=DEVICE, requires_grad=False)
    cam = cam * 0
    cam[0, 0, 100:150, 100:150] = 1
    limgs = [(cam_2Img(cam), 'CAM')]

    for ksz in [1, 3, 5, 7]:
        set_seed(seed)
        module = GetFastSeederSLFCAMS(
            min_=min_,
            max_=max_,
            min_p=min_p,
            fg_erode_k=fg_erode_k,
            fg_erode_iter=fg_erode_iter,
            ksz=ksz,
            device=DEVICE,
            support_background=True,
            multi_label_flag=False,
            seg_ignore_idx=-255)
        announce_msg('Testing {}'.format(module))
        t0 = dt.datetime.now()
        out = module(cam)
        print('time: {}'.format(dt.datetime.now() - t0))
        print(out.shape, (out == 1).sum())
        out[out == 1] = 255
        out[out == 0] = int(255 / 2)
        out[out == module.ignore_idx] = 0

        limgs.append((out.squeeze().cpu().numpy().astype(np.uint8),
                      'pseudo_ksz_{}'.format(ksz)))

    plot_limgs(limgs)


def test_GetAdaptivePseudoMaskSLFCAMS():
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap

    def get_cm():
        col_dict = dict()
        for i in range(256):
            col_dict[i] = 'k'

        col_dict[0] = 'k'
        col_dict[int(255 / 2)] = 'b'
        col_dict[255] = 'r'
        colormap = ListedColormap([col_dict[x] for x in col_dict.keys()])

        return colormap

    def cam_2Img(_cam):
        return _cam.squeeze().cpu().numpy().astype(np.uint8) * 255

    def plot_limgs(_lims):
        nrows = 1
        ncols = len(_lims)

        him, wim = _lims[0][0].shape
        r = him / float(wim)
        fw = 10
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))
        for i, (im, tag) in enumerate(_lims):
            axes[0, i].imshow(im, cmap=get_cm())
            axes[0, i].text(3, 40, tag,
                            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8})

        plt.show()

    cuda = "1"
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    seed = 0
    min_ = 1000
    max_ = 5000
    cam = torch.rand((1, 1, 224, 224), dtype=torch.float,
                     device=DEVICE, requires_grad=False)
    cam = cam * 0
    cam[0, 0, 100:150, 100:150] = 1
    limgs = [(cam_2Img(cam), 'CAM')]

    for ksz in [1, 3, 5, 7]:
        set_seed(seed)
        module = GetAdaptivePseudoMaskSLFCAMS(
            min_=min_,
            max_=max_,
            ksz=ksz,
            device=DEVICE,
            support_background=True,
            multi_label_flag=False,
            seg_ignore_idx=-255)
        announce_msg('Testing {}'.format(module))
        t0 = dt.datetime.now()
        out = module(cam)
        print('time: {}'.format(dt.datetime.now() - t0))
        print(out.shape, (out == 1).sum())
        out[out == 1] = 255
        out[out == 0] = int(255 / 2)
        out[out == module.ignore_idx] = 0

        limgs.append((out.squeeze().cpu().numpy().astype(np.uint8),
                      'pseudo_ksz_{}'.format(ksz)))

    plot_limgs(limgs)


def test_GetPseudoMaskSLFCAMS():
    cuda = "1"
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    # for ksz in [1, 3]:
    #     set_seed(1)
    #     module = GetPseudoMaskSLFCAMS(min_=2, max_=4, min_ext=100, max_ext=100,
    #                                   block=1, ksz=ksz, device=DEVICE,
    #                                   support_background=True,
    #                                   multi_label_flag=False,
    #                                   seg_ignore_idx=-255)
    #     announce_msg('Testing {}'.format(module))
    #     t0 = dt.datetime.now()
    #     out = module(cam)
    #     print('time: {}'.format(dt.datetime.now() - t0))
    #     print(out.shape, (out == 1).sum())

    print('SECOND TEST')
    for seed in range(10):
        for ksz in [1]:
            for block in [4]:
                set_seed(seed)
                cam = torch.rand((1, 1, 224, 224), dtype=torch.float,
                                 device=DEVICE,  requires_grad=False)
                module = GetPseudoMaskSLFCAMS(min_=1, max_=1, min_ext=100,
                                              max_ext=100,
                                              block=block, ksz=ksz,
                                              device=DEVICE,
                                              support_background=True,
                                              multi_label_flag=False,
                                              seg_ignore_idx=-255)
                announce_msg('Testing {}'.format(module))
                t0 = dt.datetime.now()
                out = module(cam)
                print('time: {}'.format(dt.datetime.now() - t0))
                print(out.shape, (out == 1).sum())


if __name__ == "__main__":
    import datetime as dt

    from dlib.utils.shared import announce_msg
    from dlib.utils.reproducibility import set_seed

    set_seed(0)
    # test_GetPseudoMaskSLFCAMS()
    # test_GetAdaptivePseudoMaskSLFCAMS()
    test_GetFastSeederSLFCAMS()
