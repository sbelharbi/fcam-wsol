import os
import sys
from os.path import join, dirname, abspath
import datetime as dt

import munch


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants

__all__ = ['get_config']


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def configure_data_paths(args, dsname=None):
    if dsname is None:
        dsname = args['dataset']

    train = val = test = join(args['data_root'], dsname)
    data_paths = mch(train=train, val=val, test=test)
    return data_paths


def get_root_wsol_dataset():
    baseurl = None
    if "HOST_XXX" in os.environ.keys():
        if os.environ['HOST_XXX'] == 'laptop':
            baseurl = "{}/datasets".format(os.environ["EXDRIVE"])
        elif os.environ['HOST_XXX'] == 'lab':
            baseurl = "{}/wsol-done-right".format(os.environ["DATASETSH"])
        elif os.environ['HOST_XXX'] == 'gsys':
            baseurl = "{}/wsol-done-right".format(os.environ["DATASETSH"])
        elif os.environ['HOST_XXX'] == 'ESON':
            baseurl = "{}/datasets".format(os.environ["DATASETSH"])

    elif "CC_CLUSTER" in os.environ.keys():
        if "SLURM_TMPDIR" in os.environ.keys():
            # if we are running within a job use the node disc:  $SLURM_TMPDIR
            baseurl = "{}/datasets/wsol-done-right".format(
                os.environ["SLURM_TMPDIR"])
        else:
            # if we are not running within a job, use the scratch.
            # this cate my happen if someone calls this function outside a job.
            baseurl = "{}/datasets/wsol-done-right".format(os.environ["SCRATCH"])

    msg_unknown_host = "Sorry, it seems we are enable to recognize the " \
                       "host. You seem to be new to this code. " \
                       "We recommend you to add your baseurl on your own."
    if baseurl is None:
        raise ValueError(msg_unknown_host)

    return baseurl


def get_config(ds):
    assert ds in constants.datasets

    args = {
        # ======================================================================
        #                               GENERAL
        # ======================================================================
        "MYSEED": 0,  # Seed for reproducibility. int >= 0.
        "cudaid": 0,  # int. cudaid.
        "debug_subfolder": '',  # subfolder used for debug. if '', we do not
        # consider it.
        "dataset": ds,  # name of the dataset.
        "num_classes": constants.NUMBER_CLASSES[ds],  # Total number of classes.
        "crop_size": constants.CROP_SIZE,  # int. size of cropped patch.
        "resize_size": constants.RESIZE_SIZE,  # int. size to which the image
        # is resized before cropping.
        "batch_size": 8,  # the batch size for training.
        "num_workers": 8,  # number of workers for dataloader of the trainset.
        "exp_id": "123456789",  # exp id. random number unique for the exp.
        "verbose": True,  # if true, we print messages in stdout.
        'fd_exp': None,  # relative path to folder where the exp.
        'abs_fd_exp': None,  # absolute path to folder where the exp.
        'best_epoch': 0,  # int. best epoch.
        'img_range': constants.RANGE_TANH,  # range of the image values after
        # normalization either in [0, 1] or [-1, 1]. see constants.
        't0': dt.datetime.now(),  # approximate time of starting the code.
        'tend': None,  # time when this code ends.
        'running_time': None,  # the time needed to run the entire code.
        # ======================================================================
        #                      WSOL DONE RIGHT
        # ======================================================================
        "data_root": get_root_wsol_dataset(),  # absolute path to data parent
        # folder.
        "metadata_root": constants.RELATIVE_META_ROOT,  # path to metadata.
        # contains splits.
        "mask_root": get_root_wsol_dataset(),  # path to masks.
        "proxy_training_set": False,  # efficient hyper-parameter search with
        # a proxy training set. true/false.
        "std_cams_folder": mch(train='', val='', test=''),  # folders where
        # cams of std_cl are stored to be used for f_cl training. typicaly,
        # we store only training. this is an option since f_cl can still
        # compute the std_cals. but, storing them making their access fast
        # to avoid re-computing them every time during training. the exact
        # location will be determined during parsing the input. this is
        # optional. if we do not find this folder, we recompute the cams.
        "num_val_sample_per_class": 0,  # number of full_supervision
        # validation sample per class. 0 means: use all available samples.
        'cam_curve_interval': .001,  # CAM curve interval.
        'multi_contour_eval': True,  # Bounding boxes are extracted from all
        # contours in the thresholded score map. You can use this feature by
        # setting multi_contour_eval to True (default). Otherwise,
        # bounding boxes are extracted from the largest connected
        # component of the score map.
        'multi_iou_eval': True,
        'iou_threshold_list': [30, 50, 70],
        'box_v2_metric': False,
        'eval_checkpoint_type': constants.BEST,  # just for
        # stand-alone inference. during training+inference, we evaluate both.
        # Necessary s well for the task F_CL during training to select the
        # init-model-weights-classifier.
        # ======================================================================
        #                      VISUALISATION OF REGIONS OF INTEREST
        # ======================================================================
        "alpha_visu": 100,  # transparency alpha for cams visualization. low is
        # opaque (matplotlib).
        "height_tag": 60,  # the height of the margin where the tag is written.
        # ======================================================================
        #                             OPTIMIZER (n0)
        #                            TRAIN THE MODEL
        # ======================================================================
        "optimizer": {  # the optimizer
            # ==================== SGD =======================
            "opt__name_optimizer": "sgd",  # str name. 'sgd', 'adam'
            "opt__lr": 0.001,  # Initial learning rate.
            "opt__momentum": 0.9,  # Momentum.
            "opt__dampening": 0.,  # dampening.
            "opt__weight_decay": 1e-4,  # The weight decay (L2) over the
            # parameters.
            "opt__nesterov": True,  # If True, Nesterov algorithm is used.
            # ==================== ADAM =========================
            "opt__beta1": 0.9,  # beta1.
            "opt__beta2": 0.999,  # beta2
            "opt__eps_adam": 1e-08,  # eps. for numerical stability.
            "opt__amsgrad": False,  # Use amsgrad variant or not.
            # ========== LR scheduler: how to adjust the learning rate. ========
            "opt__lr_scheduler": True,  # if true, we use a learning rate
            # scheduler.
            # ========> MyStepLR: override torch.optim.lr_scheduler.StepLR
            "opt__name_lr_scheduler": "mystep",  # str name.
            "opt__step_size": 40,  # Frequency of which to adjust the lr.
            "opt__gamma": 0.1,  # the update coefficient: lr = gamma * lr.
            "opt__last_epoch": -1,  # the index of the last epoch where to stop
            # adjusting the LR.
            "opt__min_lr": 1e-7,  # minimum allowed value for lr.
            "opt__t_max": 100,  # T_max for cosine schedule.
            "opt__lr_classifier_ratio": 10.,  # Multiplicative factor on the
            # classifier layer (head) learning rate.
        },
        # ======================================================================
        #                              MODEL
        # ======================================================================
        "model": {
            "arch": constants.UNETFCAM,  # name of the model.
            # see: constants.nets.
            "encoder_name": constants.RESNET50,  # backbone for task of SEG.
            "encoder_weights": constants.IMAGENET,
            # pretrained weights or 'None'.
            "in_channels": 3,  # number of input channel.
            "path_pre_trained": None,
            # None, `None` or a valid str-path. if str,
            # it is the absolute/relative path to the pretrained model. This can
            # be useful to resume training or to force using a filepath to some
            # pretrained weights.
            "strict": True,  # bool. Must be always be True. if True,
            # the pretrained model has to have the exact architecture as this
            # current model. if not, an error will be raise. if False, we do the
            # best. no error will be raised in case of mismatch.
            "support_background": True,  # useful for classification tasks only:
            # std_cl, f_cl only. if true, an additional cam is used for the
            # background. this does not change the number of global
            # classification logits. irrelevant for segmentation task.
            "scale_in": 1.,  # float > 0.  how much to scale
            # the input image to not overflow the memory. This scaling is done
            # inside the model on the same device as the model.
            "freeze_cl": False,  # applied only for task F_CL. if true,
            # the classifier (encoder + head) is frozen.
            "folder_pre_trained_cl": None,
            # NAME of folder containing weights of
            # classifier. it must be in in 'pretrained' folder.
            # used in combination with `freeze_cl`. the folder contains
            # encoder.pt, head.pt weights of the encoder and head. the base name
            # of the folder is a tag used to make sure of compatibility between
            # the source (source of weights) and target model (to be frozen).
            # You do not need to set this parameters if `freeze_cl` is true.
            # we set it automatically when parsing the parameters.
        },
        # ======================================================================
        #                    CLASSIFICATION SPATIAL POOLING
        # ======================================================================
        "method": constants.METHOD_WILDCAT,
        "spatial_pooling": constants.WILDCATHEAD,
        # ======================================================================
        #                        SPATIAL POOLING:
        #                            WILDCAT
        # ======================================================================
        "wc_modalities": 5,
        "wc_kmax": 0.5,
        "wc_kmin": 0.1,
        "wc_alpha": 0.6,
        "wc_dropout": 0.0,
        # ================== LSE pooling
        "lse_r": 10.,  # r for logsumexp pooling.
        # ======================================================================
        #                          Segmentation mode
        # ======================================================================
        "seg_mode": constants.BINARY_MODE,
        # SEGMENTATION mode: bin only always.
        "task": constants.STD_CL,  # task: standard classification,
        # full classification (FCAM).
        "multi_label_flag": False,
        # whether the dataset has multi-label or not.
        # ======================================================================
        #                          ELB
        # ==========================================================================
        "elb_init_t": 1.,  # used for ELB.
        "elb_max_t": 10.,  # used for ELB.
        "elb_mulcoef": 1.01,  # used for ELB.
        # ======================================================================
        #                            CONSTRAINTS:
        #                     'SuperResolution', sr
        #                     'ConRanFieldFcams', crf_fc
        #                     'EntropyFcams', entropy_fc
        #                     'PartUncerknowEntropyLowCams', partuncertentro_lc
        #                     'PartCertKnowLowCams', partcert_lc
        #                     'MinSizeNegativeLowCams', min_sizeneg_lc
        #                     'MaxSizePositiveLowCams', max_sizepos_lc
        #                     'MaxSizePositiveFcams' max_sizepos_fc
        # ======================================================================
        "max_epochs": 150,  # number of training epochs.
        # -----------------------  FCAM
        "sl_fc": False,  # use self-learning over fcams.
        "sl_fc_lambda": 1.,  # lambda for self-learning over fcams
        "sl_start_ep": 0,  # epoch when to start sl loss.
        "sl_end_ep": -1,  # epoch when to stop using sl loss. -1: never stop.
        "sl_min": 10,  # int. number of pixels to be considered
        # background (after sorting all pixels).
        "sl_max": 10,  # number of pixels to be considered
        # foreground (after sorting all pixels).
        "sl_min_ext": 1000,  # the extent of pixels to consider for selecting
        # sl_min from.
        "sl_max_ext": 1000,  # the extent of pixels to consider for selecting
        # sl_max.
        "sl_block": 1,  # size of the block. instead of selecting from pixel,
        # we allow initial selection from grid created from blocks of size
        # sl_blockxsl_block. them, for each selected block, we select a random
        # pixel. this helps selecting from fare ways regions. if you don't want
        # to use blocks, set this to 1 where the selection is done directly over
        # pixels without passing through blocks.
        "sl_ksz": 1,  # int, kernel size for dilation around the pixel. must be
        # odd number.
        'sl_min_p': .2,  # percentage of pixels to be used for background
        # sampling. percentage from entire image size.
        'sl_fg_erode_k': 11,  # int. size of erosion kernel to clean foreground.
        'sl_fg_erode_iter': 1,  # int. number of erosions for foreground.
        # ----------------------- FCAM
        "crf_fc": False,  # use or not crf over fcams.  (penalty)
        "crf_lambda": 2.e-9,  # crf lambda
        "crf_sigma_rgb": 15.,
        "crf_sigma_xy": 100.,
        "crf_scale": 1.,  # scale factor for input, segm.
        "crf_start_ep": 0,  # epoch when to start crf loss.
        "crf_end_ep": -1,  # epoch when to stop using crf loss. -1: never stop.
        # ======================================================================
        # ======================================================================
        #                                EXTRA
        # ======================================================================
        # ======================================================================
        # ----------------------- FCAM
        "entropy_fc": False,  # use or not the entropy over fcams. (penalty)
        "entropy_fc_lambda": 1.,
        # -----------------------  FCAM
        "max_sizepos_fc": False,  # use absolute size (unsupervised) over all
        # fcams. (elb)
        "max_sizepos_fc_lambda": 1.,
        "max_sizepos_fc_start_ep": 0,  # epoch when to start maxsz loss.
        "max_sizepos_fc_end_ep": -1,  # epoch when to stop using mxsz loss. -1:
        # never stop.
        # ----------------------------------------------------------------------
        # ----------------------- NOT USED
        # ------------------------------- GENERIC
        "im_rec": False,  # image reconstruction loss.
        "im_rec_lambda": 1.,
        "im_rec_elb": False,  # use or not elb for image reconstruction.
        # ----------------------------- NOT USED
        # ----------------------------------------------------------------------
        # ======================================================================
        # ======================================================================
        # ======================================================================

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ GENERIC
        'seg_ignore_idx': -255,  # ignore index for segmentation alignment.
    }

    pre = constants.FORMAT_DEBUG.split('_')[0]
    dsname = args['dataset']
    if dsname.startswith(pre):
        dsname = dsname.replace('{}_'.format(pre), '')

    args['data_paths'] = configure_data_paths(args, dsname)
    args['metadata_root'] = join(args['metadata_root'], args['dataset'])

    openimg_ds = constants.OpenImages
    if openimg_ds.startswith(pre):
        openimg_ds = dsname.replace('{}_'.format(pre), '')
    args['mask_root'] = join(args['mask_root'], openimg_ds)

    data_cams = join(root_dir, constants.DATA_CAMS)
    if not os.path.isdir(data_cams):
        os.makedirs(data_cams)

    return args


if __name__ == '__main__':
    args = get_config(constants.CUB)
    print(args['metadata_root'])
