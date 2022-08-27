import os
import sys
from os.path import dirname, abspath, join
from typing import Optional, Union
from copy import deepcopy
import pickle as pkl
import math
import datetime as dt


import numpy as np
import torch
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import yaml
import torch.nn.functional as F


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.datasets.wsol_loader import get_data_loader

from dlib.utils.reproducibility import set_seed
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.tools import get_cpu_device
from dlib.utils.tools import get_tag

from dlib.cams.selflearning import GetPseudoMaskSLFCAMS
from dlib.cams.selflearning import GetAdaptivePseudoMaskSLFCAMS
from dlib.cams.selflearning import GetFastSeederSLFCAMS
from dlib.learning.inference_wsol import CAMComputer
from dlib.cams import build_std_cam_extractor

from dlib import losses


__all__ = ['Basic', 'Trainer']


class PerformanceMeter(object):
    def __init__(self, split, higher_is_better=True):
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None
        val = constants.VALIDSET
        self.value_per_epoch = [] \
            if split == val else [-np.inf if higher_is_better else np.inf]

    def update(self, new_value):
        self.value_per_epoch.append(new_value)
        self.current_value = self.value_per_epoch[-1]
        self.best_value = self.best_function(self.value_per_epoch)
        self.best_epoch = self.value_per_epoch.index(self.best_value)


class Basic(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = (constants.TRAINSET, constants.VALIDSET, constants.TESTSET)
    _EVAL_METRICS = ['loss', 'classification', 'localization']
    # _BEST_CRITERION_METRIC = 'localization'
    _NUM_CLASSES_MAPPING = {
        constants.CUB: constants.NUMBER_CLASSES[constants.CUB],
        constants.ILSVRC: constants.NUMBER_CLASSES[constants.ILSVRC],
        constants.OpenImages: constants.NUMBER_CLASSES[constants.OpenImages],
    }

    @property
    def _BEST_CRITERION_METRIC(self):
        assert self.inited
        assert self.args is not None

        return 'localization'

        if self.args.task == constants.STD_CL:
            return 'classification'
        elif self.args.task == constants.F_CL:
            return 'localization'
        else:
            raise NotImplementedError

    def __init__(self, args):
        self.args = args
        
    def _set_performance_meters(self):
        self._EVAL_METRICS += ['localization_IOU_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]

        self._EVAL_METRICS += ['top1_loc_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]

        self._EVAL_METRICS += ['top5_loc_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]

        eval_dict = {
            split: {
                metric: PerformanceMeter(split,
                                         higher_is_better=False
                                         if metric == 'loss' else True)
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }
        return eval_dict


class Trainer(Basic):

    def __init__(self,
                 args,
                 model,
                 optimizer,
                 lr_scheduler,
                 loss: losses.MasterLoss,
                 device,
                 classifier=None):
        super(Trainer, self).__init__(args=args)

        self.device = device
        self.args = args
        self.performance_meters = self._set_performance_meters()
        self.model = model
        self.loss: losses.MasterLoss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.num_workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class,
            std_cams_folder=self.args.std_cams_folder
        )

        self.sl_mask_builder: GetFastSeederSLFCAMS = self._get_sl(
            args, device)

        self.epoch = 0
        self.counter = 0
        self.seed = int(os.environ["MYSEED"])
        self.default_seed = int(os.environ["MYSEED"])

        self.best_model = deepcopy(self.model).to(self.cpu_device).eval()
        self.last_model = deepcopy(self.model).to(self.cpu_device).eval()

        self.perf_meters_backup = None
        self.inited = True

        self.classifier = classifier
        self.std_cam_extractor = None
        if args.task == constants.F_CL:
            assert classifier is not None
            self.std_cam_extractor = self._build_std_cam_extractor(
                classifier=classifier, args=args)

        self.fcam_argmax = False
        self.fcam_argmax_previous = False

        self.t_init_epoch = dt.datetime.now()
        self.t_end_epoch = dt.datetime.now()

    @staticmethod
    def _build_std_cam_extractor(classifier, args):
        classifier.eval()
        return build_std_cam_extractor(classifier=classifier, args=args)

    def _get_sl(self, args, device):

        sl_mask_builder = GetFastSeederSLFCAMS(
            min_=args.sl_min,
            max_=args.sl_max,
            ksz=args.sl_ksz,
            min_p=args.sl_min_p,
            fg_erode_k=args.sl_fg_erode_k,
            fg_erode_iter=args.sl_fg_erode_iter,
            device=device,
            support_background=args.model['support_background'],
            multi_label_flag=args.multi_label_flag,
            seg_ignore_idx=args.seg_ignore_idx)
        return sl_mask_builder

    def fast_get_std_cam(self, cams_inter, y_global):
        assert cams_inter.shape[1] > 1
        assert cams_inter.shape[0] == 1

        index = y_global

        if self.args.model['support_background']:
            index = index + 1
        return cams_inter[0, index, :, :].unsqueeze(0).unsqueeze(0)

    def build_seeds_from_fcams(self, cams_inter, y_global) -> torch.Tensor:

        assert not self.args.multi_label_flag
        assert cams_inter.shape[0] == y_global.shape[0]

        bsz, c, h, w = cams_inter.shape
        seeds = torch.zeros((bsz, h, w), device=self.device,
                            requires_grad=False, dtype=torch.long)

        with torch.no_grad():
            for i in range(bsz):
                y = y_global[i]
                cams = cams_inter[i].unsqueeze(0)
                pulled_cam = self.fast_get_std_cam(cams, y)
                seeds[i] = self.sl_mask_builder(pulled_cam).squeeze(0)

            return seeds

    def get_std_cams_minibatch(self, images, targets,
                               std_cams: Optional[Union[None, torch.Tensor]]
                               ) -> torch.Tensor:
        # used only for task f_cl
        assert self.args.task == constants.F_CL
        assert images.ndim == 4
        image_size = images.shape[2:]

        if std_cams is not None:
            assert std_cams.ndim == 4

        cams = None
        for idx, (image, target) in enumerate(zip(images, targets)):
            if std_cams is None:
                cl_logits = self.classifier(image.unsqueeze(0))
                cam = self.std_cam_extractor(
                    class_idx=target.item(), scores=cl_logits, normalized=True)
                # todo: set to false (normalize).
            else:
                cam = std_cams[idx].squeeze()  # h, w
                # assert 0. <= cam.min() <= 1.
                # assert 0. <= cam.max() <= 1.
                assert cam.ndim == 2

            # Quick fix: todo...
            cam = torch.nan_to_num(cam, nan=0.0, posinf=1., neginf=0.0)

            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                image_size,
                                mode='bilinear',
                                align_corners=False)  # (1, 1, h, w)
            cam = cam.detach()
            if cams is None:
                cams = cam
            else:
                cams = torch.vstack((cams, cam))
        # cams: (bsz, 1, h, w)
        assert cams.ndim == 4
        return cams

    def is_seed_required(self, _epoch):
        cmd = (self.args.task == constants.F_CL)
        cmd &= ('self_learning_fcams' in self.loss.n_holder)
        cmd2 = False
        for _l in self.loss.losses:
            if isinstance(_l, losses.SelfLearningFcams):
                cmd2 = _l.is_on(_epoch=_epoch)

        return cmd and cmd2

    def _wsol_training(self, images, raw_imgs, targets, std_cams):
        y_global = targets
        output = self.model(images)

        if self.args.task == constants.STD_CL:
            cl_logits = output
            loss = self.loss(epoch=self.epoch, cl_logits=cl_logits,
                             glabel=y_global)
            logits = cl_logits

        elif self.args.task == constants.F_CL:
            cl_logits, fcams, im_recon = output

            if self.is_seed_required(_epoch=self.epoch):
                cams_inter = self.get_std_cams_minibatch(images=images,
                                                         targets=targets,
                                                         std_cams=std_cams)
                seeds = self.sl_mask_builder(cams_inter)
            else:
                cams_inter, seeds = None, None

            loss = self.loss(
                epoch=self.epoch,
                cams_inter=cams_inter,
                fcams=fcams,
                cl_logits=cl_logits,
                glabel=y_global,
                raw_img=raw_imgs,
                x_in=self.model.x_in,
                im_recon=im_recon,
                seeds=seeds
            )

            logits = cl_logits
        else:
            raise NotImplementedError

        return logits, loss

    def on_epoch_start(self):
        self.t_init_epoch = dt.datetime.now()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        self.model.train()

    def on_epoch_end(self):
        self.loss.update_t()
        # todo: temp. delete later.
        self.loss.check_losses_status()

        self.t_end_epoch = dt.datetime.now()
        delta_t = self.t_end_epoch - self.t_init_epoch
        DLLogger.log(fmsg('Train epoch runtime: {}'.format(delta_t)))

    def random(self):
        self.counter = self.counter + 1
        self.seed = self.seed + self.counter
        set_seed(seed=self.seed, verbose=False)

    def train(self, split, epoch):
        self.epoch = epoch
        self.random()
        self.on_epoch_start()

        loader = self.loaders[split]

        total_loss = 0.0
        num_correct = 0
        num_images = 0

        # todo: load pre-computed cam for f_cl if it exists.
        for batch_idx, (images, targets, _, raw_imgs, std_cams) in tqdm(
                enumerate(loader), ncols=constants.NCOLS, total=len(loader)):

            self.random()

            images = images.to(self.device)
            targets = targets.to(self.device)

            if std_cams.ndim == 1:
                std_cams = None
            else:
                assert std_cams.ndim == 4
                std_cams = std_cams.to(self.device)

            logits, loss = self._wsol_training(images, raw_imgs, targets,
                                               std_cams)
            pred = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            num_correct += (pred == targets).sum().item()
            num_images += images.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_average = total_loss / float(num_images)
        classification_acc = num_correct / float(num_images) * 100

        self.performance_meters[split]['classification'].update(
            classification_acc)
        self.performance_meters[split]['loss'].update(loss_average)

        self.on_epoch_end()

        return dict(classification_acc=classification_acc,
                    loss=loss_average)

    def print_performances(self, checkpoint_type=None):
        tagargmax = ''
        if self.fcam_argmax:
            tagargmax = ' Argmax: True'
        if checkpoint_type is not None:
            DLLogger.log(fmsg('PERF - CHECKPOINT: {} {}'.format(
                checkpoint_type, tagargmax)))

        for split in self._SPLITS:
            for metric in self._EVAL_METRICS:
                current_performance = \
                    self.performance_meters[split][metric].current_value
                if current_performance is not None:
                    DLLogger.log(
                        "Split {}, metric {}, current value: {}".format(
                         split, metric, current_performance))
                    if split != constants.TESTSET:
                        DLLogger.log(
                            "Split {}, metric {}, best value: {}".format(
                             split, metric,
                             self.performance_meters[split][metric].best_value))
                        DLLogger.log(
                            "Split {}, metric {}, best epoch: {}".format(
                             split, metric,
                             self.performance_meters[split][metric].best_epoch))

    def serialize_perf_meter(self) -> dict:
        return {
            split: {
                metric: vars(self.performance_meters[split][metric])
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }

    def save_performances(self, epoch=None, checkpoint_type=None):
        tag = '' if checkpoint_type is None else '_{}'.format(checkpoint_type)

        tagargmax = ''
        if self.fcam_argmax:
            tagargmax = '_Argmax_True'

        log_path = join(self.args.outd, 'performance_log{}{}.pickle'.format(
            tag, tagargmax))
        with open(log_path, 'wb') as f:
            pkl.dump(self.serialize_perf_meter(), f)

        log_path = join(self.args.outd, 'performance_log{}{}.txt'.format(
            tag, tagargmax))
        with open(log_path, 'w') as f:
            f.write("PERF - CHECKPOINT {}  - EPOCH {}  {} \n".format(
                checkpoint_type, epoch, tagargmax))

            for split in self._SPLITS:
                for metric in self._EVAL_METRICS:

                    f.write("REPORT EPOCH/{}: split: {}/metric {}: {} \n"
                            "".format(epoch, split, metric,
                                      self.performance_meters[split][
                                          metric].current_value))
                    f.write(
                        "REPORT EPOCH/{}: split: {}/metric {}: {}_best "
                        "\n".format(epoch, split, metric,
                                    self.performance_meters[split][
                                        metric].best_value))

    def cl_forward(self, images):

        output = self.model(images)

        if self.args.task == constants.STD_CL:
            cl_logits = output

        elif self.args.task == constants.F_CL:
            cl_logits, fcams, im_recon = output
        else:
            raise NotImplementedError

        return cl_logits

    def _compute_accuracy(self, loader):
        num_correct = 0
        num_images = 0

        for i, (images, targets, _, _, _) in enumerate(loader):
            images = images.cuda()
            targets = targets.cuda()
            with torch.no_grad():
                cl_logits = self.cl_forward(images)
                pred = cl_logits.argmax(dim=1)

            num_correct += (pred == targets).sum().item()
            num_images += images.size(0)

        classification_acc = num_correct / float(num_images) * 100
        return classification_acc

    def evaluate(self, epoch, split, checkpoint_type=None, fcam_argmax=False):
        if fcam_argmax:
            assert self.args.task == constants.F_CL

        self.fcam_argmax_previous = self.fcam_argmax
        self.fcam_argmax = fcam_argmax
        tagargmax = ''
        if self.args.task == constants.F_CL:
            tagargmax = 'Argmax {}'.format(fcam_argmax)

        DLLogger.log(fmsg("Evaluate: Epoch {} Split {} {}".format(
            epoch, split, tagargmax)))

        outd = None
        if split == constants.TESTSET:
            assert checkpoint_type is not None
            if fcam_argmax:
                outd = join(self.args.outd, checkpoint_type, 'argmax-true',
                            split)
            else:
                outd = join(self.args.outd, checkpoint_type, split)
            if not os.path.isdir(outd):
                os.makedirs(outd)

        set_seed(seed=self.default_seed, verbose=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model.eval()

        accuracy = self._compute_accuracy(loader=self.loaders[split])
        self.performance_meters[split]['classification'].update(accuracy)

        cam_curve_interval = self.args.cam_curve_interval
        cmdx = (split == constants.VALIDSET)
        cmdx &= self.args.dataset in [constants.CUB, constants.ILSVRC]
        if cmdx:
            cam_curve_interval = constants.VALID_FAST_CAM_CURVE_INTERVAL

        cam_computer = CAMComputer(
            args=deepcopy(self.args),
            model=self.model,
            loader=self.loaders[split],
            metadata_root=os.path.join(self.args.metadata_root, split),
            mask_root=self.args.mask_root,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset,
            split=split,
            cam_curve_interval=cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            out_folder=outd,
            fcam_argmax=fcam_argmax
        )
        t0 = dt.datetime.now()

        cam_performance = cam_computer.compute_and_evaluate_cams()

        DLLogger.log(fmsg("CAM EVALUATE TIME of {} split: {}".format(
            split, dt.datetime.now() - t0)))

        if split == constants.TESTSET:
            cam_computer.draw_some_best_pred()

        if self.args.multi_iou_eval or (self.args.dataset ==
                                        constants.OpenImages):
            loc_score = np.average(cam_performance)
        else:
            loc_score = cam_performance[self.args.iou_threshold_list.index(50)]

        self.performance_meters[split]['localization'].update(loc_score)

        if self.args.dataset in (constants.CUB, constants.ILSVRC):
            for idx, IOU_THRESHOLD in enumerate(self.args.iou_threshold_list):
                self.performance_meters[split][
                    'localization_IOU_{}'.format(IOU_THRESHOLD)].update(
                    cam_performance[idx])

                self.performance_meters[split][
                    'top1_loc_{}'.format(IOU_THRESHOLD)].update(
                    cam_computer.evaluator.top1[idx])

                self.performance_meters[split][
                    'top5_loc_{}'.format(IOU_THRESHOLD)].update(
                    cam_computer.evaluator.top5[idx])

            if split == constants.TESTSET:
                curve_top_1_5 = cam_computer.evaluator.curve_top_1_5
                with open(join(outd, 'curves_top_1_5.pkl'), 'wb') as fc:
                    pkl.dump(curve_top_1_5, fc, protocol=pkl.HIGHEST_PROTOCOL)

                title = get_tag(self.args, checkpoint_type=checkpoint_type)
                title = 'Top1/5: {}'.format(title)

                if fcam_argmax:
                    title += '_argmax_true'
                else:
                    title += '_argmax_false'
                self.plot_perf_curves_top_1_5(curves=curve_top_1_5, fdout=outd,
                                              title=title)

        if split == constants.TESTSET:

            curves = cam_computer.evaluator.curve_s
            with open(join(outd, 'curves.pkl'), 'wb') as fc:
                pkl.dump(curves, fc, protocol=pkl.HIGHEST_PROTOCOL)

            title = get_tag(self.args, checkpoint_type=checkpoint_type)

            if fcam_argmax:
                title += '_argmax_true'
            else:
                title += '_argmax_false'
            self.plot_perf_curves(curves=curves, fdout=outd, title=title)

    def plot_perf_curves_top_1_5(self, curves: dict, fdout: str, title: str):

        x_label = r'$\tau$'
        y_label = 'BoxAcc'

        fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)

        for i, top in enumerate(['top1', 'top5']):

            iouthres = sorted(list(curves[top].keys()))
            for iout in iouthres:
                axes[0, i].plot(curves['x'], curves[top][iout],
                                label=r'{}: $\sigma$={}'.format(top, iout))

            axes[0, i].xaxis.set_tick_params(labelsize=5)
            axes[0, i].yaxis.set_tick_params(labelsize=5)
            axes[0, i].set_xlabel(x_label, fontsize=8)
            axes[0, i].set_ylabel(y_label, fontsize=8)
            axes[0, i].grid(True)
            axes[0, i].legend(loc='best')
            axes[0, i].set_title(top)

        fig.suptitle(title, fontsize=8)
        plt.tight_layout()
        plt.show()
        fig.savefig(join(fdout, 'curves_top1_5.png'), bbox_inches='tight',
                    dpi=300)

    def plot_perf_curves(self, curves: dict, fdout: str, title: str):

        bbox = True
        x_label = r'$\tau$'
        y_label = 'BoxAcc'
        if 'y' in curves:
            bbox = False
            x_label = 'Recall'
            y_label = 'Precision'

        fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True)

        if bbox:
            iouthres = sorted([kk for kk in curves.keys() if kk != 'x'])
            for iout in iouthres:
                ax.plot(curves['x'], curves[iout],
                        label=r'$\sigma$={}'.format(iout))
        else:
            ax.plot(curves['x'], curves['y'], color='tab:orange',
                    label='Precision/Recall')

        ax.xaxis.set_tick_params(labelsize=5)
        ax.yaxis.set_tick_params(labelsize=5)
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.grid(True)
        plt.legend(loc='best')
        fig.suptitle(title, fontsize=8)
        plt.tight_layout()
        plt.show()
        fig.savefig(join(fdout, 'curves_perf.png'), bbox_inches='tight',
                    dpi=300)

    def capture_perf_meters(self):
        self.perf_meters_backup = deepcopy(self.performance_meters)

    def switch_perf_meter_to_captured(self):
        self.performance_meters = deepcopy(self.perf_meters_backup)
        self.fcam_argmax = self.fcam_argmax_previous

    def save_args(self):
        self._save_args(path=join(self.args.outd, 'config_obj_final.yaml'))

    def _save_args(self, path):
        _path = path
        with open(_path, 'w') as f:
            self.args.tend = dt.datetime.now()
            yaml.dump(vars(self.args), f)

    @property
    def cpu_device(self):
        return get_cpu_device()

    def save_checkpoints(self, split):
        best_epoch = self.performance_meters[split][
            self._BEST_CRITERION_METRIC].best_epoch

        self.args.best_epoch = best_epoch
        self._save_model(checkpoint_type=constants.BEST, epoch=best_epoch)

        max_epoch = self.args.max_epochs
        self._save_model(checkpoint_type=constants.LAST, epoch=max_epoch)

    def _save_model(self, checkpoint_type, epoch):
        assert checkpoint_type in [constants.BEST, constants.LAST]

        if checkpoint_type == constants.BEST:
            _model = deepcopy(self.best_model).to(self.cpu_device).eval()
        elif checkpoint_type == constants.LAST:
            _model = deepcopy(self.last_model).to(self.cpu_device).eval()
        else:
            raise NotImplementedError

        tag = get_tag(self.args, checkpoint_type=checkpoint_type)
        path = join(self.args.outd, tag)
        if not os.path.isdir(path):
            os.makedirs(path)

        if self.args.task == constants.STD_CL:
            torch.save(_model.encoder.state_dict(), join(path, 'encoder.pt'))
            torch.save(_model.classification_head.state_dict(),
                       join(path, 'classification_head.pt'))

        elif self.args.task == constants.F_CL:
            torch.save(_model.encoder.state_dict(), join(path, 'encoder.pt'))
            torch.save(_model.decoder.state_dict(), join(path, 'decoder.pt'))
            torch.save(_model.segmentation_head.state_dict(),
                       join(path, 'segmentation_head.pt'))
            if _model.reconstruction_head is not None:
                torch.save(_model.reconstruction_head.state_dict(),
                           join(path, 'reconstruction_head.pt'))

        else:
            raise NotImplementedError

        self._save_args(path=join(path, 'config_model.yaml'))
        DLLogger.log(message="Stored Model [CP: {} \t EPOCH: {} \t TAG: {}]:"
                             " {}".format(checkpoint_type, epoch, tag, path))

    def model_selection(self, epoch, split):
        if (self.performance_meters[split][self._BEST_CRITERION_METRIC]
                .best_epoch) == epoch:
            self.best_model = deepcopy(self.model).to(self.cpu_device).eval()

        if self.args.max_epochs == epoch:
            self.last_model = deepcopy(self.model).to(self.cpu_device).eval()

    def load_checkpoint(self, checkpoint_type):
        assert checkpoint_type in [constants.BEST, constants.LAST]
        tag = get_tag(self.args, checkpoint_type=checkpoint_type)
        path = join(self.args.outd, tag)

        if self.args.task == constants.STD_CL:
            weights = torch.load(join(path, 'encoder.pt'),
                                 map_location=self.device)
            self.model.encoder.super_load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'classification_head.pt'),
                                 map_location=self.device)
            self.model.classification_head.load_state_dict(weights,
                                                           strict=True)

        elif self.args.task == constants.F_CL:
            weights = torch.load(join(path, 'encoder.pt'),
                                 map_location=self.device)
            self.model.encoder.super_load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'decoder.pt'),
                                 map_location=self.device)
            self.model.decoder.load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'segmentation_head.pt'),
                                 map_location=self.device)
            self.model.segmentation_head.load_state_dict(weights, strict=True)

            if self.model.reconstruction_head is not None:
                weights = torch.load(join(path, 'reconstruction_head.pt'),
                                     map_location=self.device)
                self.model.reconstruction_head.load_state_dict(weights,
                                                               strict=True)
        else:
            raise NotImplementedError

        DLLogger.log("Checkpoint {} loaded.".format(path))

    def report_train(self, train_performance, epoch, split=constants.TRAINSET):
        DLLogger.log('REPORT EPOCH/{}: {}/classification: {}'.format(
            epoch, split, train_performance['classification_acc']))
        DLLogger.log('REPORT EPOCH/{}: {}/loss: {}'.format(
            epoch, split, train_performance['loss']))

    def report(self, epoch, split, checkpoint_type=None):
        tagargmax = ''
        if self.fcam_argmax:
            tagargmax = ' Argmax: True'
        if checkpoint_type is not None:
            DLLogger.log(fmsg('PERF - CHECKPOINT: {} {}'.format(
                checkpoint_type, tagargmax)))

        for metric in self._EVAL_METRICS:
            DLLogger.log("REPORT EPOCH/{}: split: {}/metric {}: {} ".format(
                epoch, split, metric,
                self.performance_meters[split][metric].current_value))
            DLLogger.log("REPORT EPOCH/{}: split: {}/metric {}: "
                         "{}_best ".format(
                          epoch, split, metric,
                          self.performance_meters[split][metric].best_value))

    def adjust_learning_rate(self):
        self.lr_scheduler.step()

    def plot_meter(self, metrics: dict, filename: str, title: str = '',
                   xlabel: str = '', best_iter: int = None):

        ncols = 4
        ks = list(metrics.keys())
        if len(ks) > ncols:
            nrows = math.ceil(len(ks) / float(ncols))
        else:
            nrows = 1
            ncols = len(ks)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False)
        t = 0
        for i in range(nrows):
            for j in range(ncols):
                if t >= len(ks):
                    axes[i, j].set_visible(False)
                    t += 1
                    continue

                val = metrics[ks[t]]['value_per_epoch']
                x = list(range(len(val)))
                axes[i, j].plot(x, val, color='tab:orange')
                axes[i, j].set_title(ks[t], fontsize=4)
                axes[i, j].xaxis.set_tick_params(labelsize=4)
                axes[i, j].yaxis.set_tick_params(labelsize=4)
                axes[i, j].set_xlabel('#{}'.format(xlabel), fontsize=4)
                axes[i, j].grid(True)
                # axes[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))

                if best_iter is not None:
                    axes[i, j].plot([x[best_iter]], [val[best_iter]],
                                    marker='o',
                                    markersize=5,
                                    color="red")
                t += 1

        fig.suptitle(title, fontsize=4)
        plt.tight_layout()

        fig.savefig(join(self.args.outd, '{}.png'.format(filename)),
                    bbox_inches='tight', dpi=300)

    def clean_metrics(self, metric: dict) -> dict:
        _metric = deepcopy(metric)
        l = []
        for k in _metric.keys():
            cd = (_metric[k]['value_per_epoch'] == [])
            cd |= (_metric[k]['value_per_epoch'] == [np.inf])
            cd |= (_metric[k]['value_per_epoch'] == [-np.inf])

            if cd:
                l.append(k)

        for k in l:
            _metric.pop(k, None)

        return _metric

    def plot_perfs_meter(self):
        meters = self.serialize_perf_meter()
        xlabel = 'epochs'

        best_epoch = self.performance_meters[constants.VALIDSET][
            self._BEST_CRITERION_METRIC].best_epoch

        for split in [constants.TRAINSET, constants.VALIDSET]:
            title = 'DS: {}, Split: {}, box_v2_metric: {}. Best iter.:' \
                    '{} {}'.format(
                     self.args.dataset, split, self.args.box_v2_metric,
                     best_epoch, xlabel)
            filename = '{}-{}-boxv2-{}'.format(
                self.args.dataset, split, self.args.box_v2_metric)
            self.plot_meter(
                self.clean_metrics(meters[split]), filename=filename,
                title=title, xlabel=xlabel, best_iter=best_epoch)

