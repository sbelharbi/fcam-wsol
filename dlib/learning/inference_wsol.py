from pathlib import Path
import subprocess

import kornia.morphology
import numpy as np
import os
import sys
from os.path import dirname, abspath, join
import datetime as dt
import pickle as pkl
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm as tqdm

from skimage.filters import threshold_otsu
from skimage import filters

from PIL import Image

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.metrics.wsol_metrics import BoxEvaluator
from dlib.metrics.wsol_metrics import MaskEvaluator
from dlib.metrics.wsol_metrics import compute_bboxes_from_scoremaps
from dlib.metrics.wsol_metrics import calculate_multiple_iou
from dlib.metrics.wsol_metrics import get_mask
from dlib.metrics.wsol_metrics import load_mask_image

from dlib.datasets.wsol_loader import configure_metadata
from dlib.visualization.vision_wsol import Viz_WSOL

from dlib.utils.tools import t2n
from dlib.utils.tools import check_scoremap_validity
from dlib.configure import constants
from dlib.cams import build_std_cam_extractor
from dlib.cams import build_fcam_extractor
from dlib.utils.shared import reformat_id

from dlib.utils.reproducibility import set_seed
import dlib.dllogger as DLLogger


_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = constants.CROP_SIZE  # 224


def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float).
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()

    return cam


def max_normalize(cam):
    max_val = cam.max()
    if max_val == 0.:
        return cam

    return cam / max_val


def entropy_cam(cam: torch.Tensor) -> torch.Tensor:
    assert isinstance(cam, torch.Tensor)
    assert cam.ndim == 2

    ops = 1. - cam
    entrop = - cam * torch.log2(cam) - ops * torch.log2(ops)
    assert ((entrop > 1.) + (entrop < 0.)).sum() == 0.

    return entrop


class CAMComputer(object):
    def __init__(self,
                 args,
                 model,
                 loader: DataLoader,
                 metadata_root,
                 mask_root,
                 iou_threshold_list,
                 dataset_name,
                 split,
                 multi_contour_eval,
                 cam_curve_interval: float = .001,
                 out_folder=None,
                 fcam_argmax: bool = False):
        self.args = args
        self.model = model
        self.model.eval()
        self.loader = loader
        self.dataset_name = dataset_name
        self.split = split
        self.out_folder = out_folder
        self.fcam_argmax = fcam_argmax

        metadata = configure_metadata(metadata_root)
        cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))

        self.evaluator = {constants.OpenImages: MaskEvaluator,
                          constants.CUB: BoxEvaluator,
                          constants.ILSVRC: BoxEvaluator
                          }[dataset_name](metadata=metadata,
                                          dataset_name=dataset_name,
                                          split=split,
                                          cam_threshold_list=cam_threshold_list,
                                          iou_threshold_list=iou_threshold_list,
                                          mask_root=mask_root,
                                          multi_contour_eval=multi_contour_eval)

        self.viz = Viz_WSOL()
        self.default_seed = int(os.environ["MYSEED"])

        self.std_cam_extractor = None
        self.fcam_extractor = None

        if args.task == constants.STD_CL:
            self.std_cam_extractor = self._build_std_cam_extractor(
                classifier=self.model, args=self.args)
        elif args.task == constants.F_CL:
            self.fcam_extractor = self._build_fcam_extractor(
                model=self.model, args=self.args)
            # useful for drawing side-by-side.
            # todo: build classifier from scratch and create its cam extractor.
        else:
            raise NotImplementedError

    def _build_std_cam_extractor(self, classifier, args):
        return build_std_cam_extractor(classifier=classifier, args=args)

    def _build_fcam_extractor(self, model, args):
        return build_fcam_extractor(model=model, args=args)

    def extract_right_cams(self, cams, targets) -> torch.Tensor:
        assert not self.args.multi_label_flag
        task = self.args.task
        support_background = self.args.model['support_background']

        if task in [constants.STD_CL, constants.F_CL]:
            assert isinstance(cams, torch.Tensor)
            assert cams.ndim == 4
        else:
            raise NotImplementedError

        if task == constants.STD_CL:
            b = cams.shape[0]
            out = None
            for i in range(b):
                index = targets[i]

                if support_background:
                    index = index + 1

                if out is None:
                     out = cams[i, index, :, :].unsqueeze(0).unsqueeze(0)
                else:
                    out = torch.vstack(
                        (out, cams[i, index, :, :].unsqueeze(0).unsqueeze(0)))

            return out

        if task == constants.F_CL:
            cams_n = torch.softmax(cams, dim=1)
            return cams_n[:, 1, :, :].unsqueeze(1)

        raise NotImplementedError

    def get_cam_one_sample(self, image: torch.Tensor, target: int,
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.model(image)

        if self.args.task == constants.STD_CL:
            cl_logits = output
            cam = self.std_cam_extractor(class_idx=target,
                                         scores=cl_logits,
                                         normalized=True)

            # (h`, w`)

        elif self.args.task == constants.F_CL:
            cl_logits, fcams, im_recon = output
            cam = self.fcam_extractor(argmax=self.fcam_argmax)
            # (h`, w`)

        else:
            raise NotImplementedError

        # Quick fix: todo...
        cam = torch.nan_to_num(cam, nan=0.0, posinf=1., neginf=0.0)
        # cl_logits: 1, nc.
        return cam, cl_logits

    def minibatch_accum(self, images, targets, image_ids, image_size) -> None:

        for image, target, image_id in zip(images, targets, image_ids):
            cam, cl_logits = self.get_cam_one_sample(
                image=image.unsqueeze(0), target=target.item())
            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                image_size,
                                mode='bilinear',
                                align_corners=False).squeeze(0).squeeze(0)
            cam = cam.detach()
            # todo:
            # cam = torch.clamp(cam, min=0.0, max=1.)

            # cam: (h, w)
            cam = t2n(cam)
            assert cl_logits.ndim == 2
            _, preds_ordered = torch.sort(input=cl_logits.cpu().squeeze(0),
                                          descending=True, stable=True)

            self.evaluator.accumulate(cam, image_id, target.item(),
                                      preds_ordered.numpy())

    def normalizecam(self, cam):
        if self.args.task == constants.STD_CL:
            cam_normalized = normalize_scoremap(cam)
        elif self.args.task == constants.F_CL:
            cam_normalized = cam
        else:
            raise NotImplementedError
        return cam_normalized

    def fix_random(self):
        set_seed(seed=self.default_seed, verbose=False)
        torch.backends.cudnn.benchmark = True

        torch.backends.cudnn.deterministic = True

    def compute_and_evaluate_cams(self):
        print("Computing and evaluating cams.")
        for batch_idx, (images, targets, image_ids, _, _) in tqdm(enumerate(
                self.loader), ncols=constants.NCOLS, total=len(self.loader)):

            self.fix_random()

            image_size = images.shape[2:]
            images = images.cuda()

            self.minibatch_accum(images=images, targets=targets,
                                 image_ids=image_ids, image_size=image_size)

            # # cams shape (batchsize, h, w)..
            # for cam, image_id in zip(cams, image_ids):
            #     # cams shape (h, w).
            #     assert cam.shape == image_size
            #
            #     # cam_resized = cv2.resize(cam, image_size,
            #     #                          interpolation=cv2.INTER_CUBIC)
            #
            #     cam_resized = cam
            #     cam_normalized = self.normalizecam(cam_resized)
            #     self.evaluator.accumulate(cam_normalized, image_id)

        return self.evaluator.compute()

    def build_bbox(self, scoremap, image_id, tau: float):
        cam_threshold_list = [tau]

        boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
            scoremap=scoremap,
            scoremap_threshold_list=cam_threshold_list,
            multi_contour_eval=self.evaluator.multi_contour_eval)

        assert len(boxes_at_thresholds) == 1
        assert len(number_of_box_list) == 1

        # nbrbox, 4
        boxes_at_thresholds = np.concatenate(boxes_at_thresholds, axis=0)

        multiple_iou = calculate_multiple_iou(
            np.array(boxes_at_thresholds),
            np.array(self.evaluator.gt_bboxes[image_id]))  # (nbrbox, 1)

        multiple_iou = multiple_iou.flatten()
        idx = np.argmax(multiple_iou)
        bbox_iou = multiple_iou[idx]
        best_bbox = boxes_at_thresholds[idx]  # shape: (4,)

        return best_bbox, bbox_iou

    def build_mask(self):
        pass

    def assert_datatset_bbx(self):
        assert self.dataset_name in [constants.CUB, constants.ILSVRC]

    def assert_dataset_mask(self):
        assert self.dataset_name == constants.OpenImages

    def assert_tau_list(self):
        iou_threshold_list = self.evaluator.iou_threshold_list
        best_tau_list = self.evaluator.best_tau_list

        if isinstance(self.evaluator, BoxEvaluator):
            assert len(best_tau_list) == len(iou_threshold_list)
        elif isinstance(self.evaluator, MaskEvaluator):
            assert len(best_tau_list) == 1
        else:
            raise NotImplementedError

    def create_folder(self, fd):
        if not os.path.isdir(fd):
            os.makedirs(fd)

    def reformat_id(self, img_id):
        tmp = str(Path(img_id).with_suffix(''))
        return tmp.replace('/', '_')

    def get_ids_with_zero_ignore_mask(self):
        ids = self.loader.dataset.image_ids

        out = []
        for id in ids:
            ignore_file = os.path.join(self.evaluator.mask_root,
                                       self.evaluator.ignore_paths[id])
            ignore_box_mask = load_mask_image(ignore_file,
                                              (_RESIZE_LENGTH, _RESIZE_LENGTH))
            if ignore_box_mask.sum() == 0:
                out.append(id)

        return out

    def select_random_ids_to_draw(self, nbr: int) -> list:
        self.fix_random()
        if isinstance(self.evaluator, BoxEvaluator):
            ids = self.loader.dataset.image_ids
            total_s = len(ids)
            n = min(nbr, total_s)
            idx = np.random.choice(a=total_s, size=n, replace=False).flatten()

        elif isinstance(self.evaluator, MaskEvaluator):
            ids = self.get_ids_with_zero_ignore_mask()
            total_s = len(ids)
            n = min(nbr, total_s)
            idx = np.random.choice(a=total_s, size=n, replace=False).flatten()
        else:
            raise NotImplementedError

        selected_ids = [ids[z] for z in idx]
        self.fix_random()

        return selected_ids

    def draw_some_best_pred(self, nbr=200, separate=False, compress=True,
                            store_imgs=False, store_cams_alone=False):
        print('Drawing some pictures')
        assert self.evaluator.best_tau_list != []
        iou_threshold_list = self.evaluator.iou_threshold_list
        best_tau_list = self.evaluator.best_tau_list
        self.assert_tau_list()

        ids_to_draw = self.select_random_ids_to_draw(nbr=nbr)
        # todo: optimize. unnecessary loading of useless samples.
        for idxb, (images, targets, image_ids, raw_imgs, _) in tqdm(enumerate(
                self.loader), ncols=constants.NCOLS, total=len(self.loader)):

            self.fix_random()

            image_size = images.shape[2:]
            images = images.cuda()

            # cams shape (batchsize, h, w).
            for image, target, image_id, raw_img in zip(
                    images, targets, image_ids, raw_imgs):
                if image_id not in ids_to_draw:
                    continue

                # raw_img: 3, h, w
                raw_img = raw_img.permute(1, 2, 0).numpy()  # h, w, 3
                raw_img = raw_img.astype(np.uint8)

                if store_imgs:
                    img_fd = join(self.out_folder, 'vizu/imgs')
                    self.create_folder(img_fd)
                    Image.fromarray(raw_img).save(join(img_fd, '{}.png'.format(
                        self.reformat_id(image_id))))

                low_cam, _ = self.get_cam_one_sample(image=image.unsqueeze(0),
                                                     target=target.item())

                cam = F.interpolate(low_cam.unsqueeze(0).unsqueeze(0),
                                    image_size,
                                    mode='bilinear',
                                    align_corners=False).squeeze(0).squeeze(0)

                cam = torch.clamp(cam, min=0.0, max=1.)

                if store_cams_alone:
                    calone_fd = join(self.out_folder, 'vizu/cams_alone/low_res')
                    self.create_folder(calone_fd)

                    self.viz.plot_cam_raw(t2n(low_cam), outf=join(
                        calone_fd, '{}.png'.format(self.reformat_id(
                            image_id))), interpolation='none')

                    calone_fd = join(self.out_folder,
                                     'vizu/cams_alone/high_res')
                    self.create_folder(calone_fd)

                    self.viz.plot_cam_raw(t2n(cam), outf=join(
                        calone_fd, '{}.png'.format(self.reformat_id(
                            image_id))), interpolation='bilinear')

                cam = torch.clamp(cam, min=0.0, max=1.)
                cam = t2n(cam)

                # cams shape (h, w).
                assert cam.shape == image_size

                cam_resized = cam
                cam_normalized = cam_resized
                check_scoremap_validity(cam_normalized)

                if isinstance(self.evaluator, BoxEvaluator):
                    self.assert_datatset_bbx()
                    l_datum = []
                    for k, _THRESHOLD in enumerate(iou_threshold_list):
                        th_fd = join(self.out_folder, 'vizu', str(_THRESHOLD))
                        self.create_folder(th_fd)

                        tau = best_tau_list[k]
                        best_bbox, bbox_iou = self.build_bbox(
                            scoremap=cam_normalized, image_id=image_id,
                            tau=tau
                        )
                        gt_bbx = self.evaluator.gt_bboxes[image_id]
                        gt_bbx = np.array(gt_bbx)
                        datum = {'img': raw_img, 'img_id': image_id,
                                 'gt_bbox': gt_bbx,
                                 'pred_bbox': best_bbox.reshape((1, 4)),
                                 'iou': bbox_iou, 'tau': tau,
                                 'sigma': _THRESHOLD, 'cam': cam_normalized}

                        if separate:
                            outf = join(th_fd, '{}.png'.format(self.reformat_id(
                                image_id)))
                            self.viz.plot_single(datum=datum, outf=outf)
                        l_datum.append(datum)

                    th_fd = join(self.out_folder, 'vizu', 'all_taux')
                    self.create_folder(th_fd)
                    outf = join(th_fd, '{}.png'.format(self.reformat_id(
                        image_id)))
                    self.viz.plot_multiple(data=l_datum, outf=outf)

                elif isinstance(self.evaluator, MaskEvaluator):
                    self.assert_dataset_mask()
                    tau = best_tau_list[0]
                    taux = sorted(list({0.5, 0.6, 0.7, 0.8, 0.9}))
                    gt_mask = get_mask(self.evaluator.mask_root,
                                       self.evaluator.mask_paths[image_id],
                                       self.evaluator.ignore_paths[image_id])
                    # gt_mask numpy.ndarray(size=(224, 224), dtype=np.uint8)

                    l_datum = []
                    for tau in taux:
                        th_fd = join(self.out_folder, 'vizu', str(tau))
                        self.create_folder(th_fd)
                        l_datum.append(
                            {'img': raw_img, 'img_id': image_id,
                             'gt_mask': gt_mask, 'tau': tau,
                             'best_tau': tau == best_tau_list[0],
                             'cam': cam_normalized}
                        )
                        # todo: plotting singles is not necessary for now.
                        # todo: control it latter for standalone inference.
                        if separate:
                            outf = join(th_fd, '{}.png'.format(self.reformat_id(
                                image_id)))
                            self.viz.plot_single(datum=l_datum[-1], outf=outf)

                    th_fd = join(self.out_folder, 'vizu', 'some_taux')
                    self.create_folder(th_fd)
                    outf = join(th_fd, '{}.png'.format(self.reformat_id(
                        image_id)))
                    self.viz.plot_multiple(data=l_datum, outf=outf)
                else:
                    raise NotImplementedError

        if compress:
            self.compress_fdout(self.out_folder, 'vizu')

    def compress_fdout(self, parent_fd, fd_trg):
        assert os.path.isdir(join(parent_fd, fd_trg))

        cmdx = [
            "cd {} ".format(parent_fd),
            "tar -cf {}.tar.gz {} ".format(fd_trg, fd_trg),
            "rm -r {} ".format(fd_trg)
        ]
        cmdx = " && ".join(cmdx)
        DLLogger.log("Running: {}".format(cmdx))
        try:
            subprocess.run(cmdx, shell=True, check=True)
        except subprocess.SubprocessError as e:
            DLLogger.log("Failed to run: {}. Error: {}".format(cmdx, e))

