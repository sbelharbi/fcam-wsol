# Sel-contained-as-possible module handles parsing the input using argparse.
# handles seed, and initializes some modules for reproducibility.

import os
from os.path import dirname, abspath, join, basename, expanduser, normpath
import sys
import argparse
from copy import deepcopy
import warnings
import subprocess
import fnmatch
import glob
import shutil
import datetime as dt

import yaml
import munch
import numpy as np

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.configure import config
from dlib.utils import reproducibility

from dlib.dllogger import ArbJSONStreamBackend
from dlib.dllogger import Verbosity
from dlib.dllogger import ArbStdOutBackend
from dlib.dllogger import ArbTextStreamBackend
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.shared import is_cc
from dlib.utils.tools import get_tag


def mkdir(fd):
    if not os.path.isdir(fd):
        os.makedirs(fd)


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def find_files_pattern(fd_in_, pattern_):
    assert os.path.exists(fd_in_), "Folder {} does not exist.".format(fd_in_)
    files = []
    for r, d, f in os.walk(fd_in_):
        for file in f:
            if fnmatch.fnmatch(file, pattern_):
                files.append(os.path.join(r, file))

    return files


def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Dict2Obj(object):
    def __init__(self, dictionary):
        for key in dictionary.keys():
            setattr(self, key, dictionary[key])

    def __repr__(self):
        attrs = str([x for x in self.__dict__])
        return "<Dict2Obj: %s" % attrs


def get_args(args: dict, eval=False):
    parser = argparse.ArgumentParser()

    parser.add_argument("--cudaid", type=str, default=None, help="cuda id.")
    parser.add_argument("--MYSEED", type=str, default=None, help="Seed.")
    parser.add_argument("--debug_subfolder", type=str, default=None,
                        help="Name of subfold for debugging. Default: ''.")

    parser.add_argument("--dataset", type=str, default=None,
                        help="Name of the dataset.")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes in the dataset.")

    parser.add_argument("--crop_size", type=int, default=None,
                        help="Crop size (int) of the patches in training.")
    parser.add_argument("--resize_size", type=int, default=None,
                        help="Resize image into this size before processing.")

    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Max epoch.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Training batch size (optimizer).")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of workers for dataloader multi-proc.")
    parser.add_argument("--exp_id", type=str, default=None, help="Exp id.")
    parser.add_argument("--verbose", type=str2bool, default=None,
                        help="Verbosity (bool).")
    parser.add_argument("--fd_exp", type=str, default=None,
                        help="Relative path to exp folder.")

    # ======================================================================
    #                      WSOL
    # ======================================================================
    parser.add_argument('--data_root', default=None,
                        help='path to dataset images')
    parser.add_argument('--metadata_root', type=str, default=None)
    parser.add_argument('--mask_root', default=None,
                        help='path to masks')
    parser.add_argument('--proxy_training_set', type=str2bool,
                        default=None,
                        help='Efficient hyper_parameter search with a proxy '
                             'training set.')
    parser.add_argument('--num_val_sample_per_class', type=int, default=None,
                        help='Number of full_supervision validation sample per '
                             'class. 0 means "use all available samples".')
    parser.add_argument('--cam_curve_interval', type=float, default=None,
                        help='CAM curve interval')
    parser.add_argument('--multi_contour_eval', type=str2bool, default=None)
    parser.add_argument('--multi_iou_eval', type=str2bool, default=None)
    parser.add_argument('--box_v2_metric', type=str2bool, default=None)
    parser.add_argument('--eval_checkpoint_type', type=str, default=None)
    # ======================================================================
    #                      OPTIMIZER
    # ======================================================================
    # opt0: optimizer for the model.
    parser.add_argument("--opt__name_optimizer", type=str, default=None,
                        help="Name of the optimizer 'sgd', 'adam'.")
    parser.add_argument("--opt__lr", type=float, default=None,
                        help="Learning rate (optimizer)")
    parser.add_argument("--opt__momentum", type=float, default=None,
                        help="Momentum (optimizer)")
    parser.add_argument("--opt__dampening", type=float, default=None,
                        help="Dampening for Momentum (optimizer)")
    parser.add_argument("--opt__nesterov", type=str2bool, default=None,
                        help="Nesterov or not for Momentum (optimizer)")
    parser.add_argument("--opt__weight_decay", type=float, default=None,
                        help="Weight decay (optimizer)")
    parser.add_argument("--opt__beta1", type=float, default=None,
                        help="Beta1 for adam (optimizer)")
    parser.add_argument("--opt__beta2", type=float, default=None,
                        help="Beta2 for adam (optimizer)")
    parser.add_argument("--opt__eps_adam", type=float, default=None,
                        help="eps for adam (optimizer)")
    parser.add_argument("--opt__amsgrad", type=str2bool, default=None,
                        help="amsgrad for adam (optimizer)")
    parser.add_argument("--opt__lr_scheduler", type=str2bool, default=None,
                        help="Whether to use or not a lr scheduler")
    parser.add_argument("--opt__name_lr_scheduler", type=str, default=None,
                        help="Name of the lr scheduler")
    parser.add_argument("--opt__gamma", type=float, default=None,
                        help="Gamma of the lr scheduler. (mystep)")
    parser.add_argument("--opt__last_epoch", type=int, default=None,
                        help="Index last epoch to stop adjust LR(mystep)")
    parser.add_argument("--opt__min_lr", type=float, default=None,
                        help="Minimum allowed value for lr.")
    parser.add_argument("--opt__t_max", type=float, default=None,
                        help="T_max, maximum epochs to restart. (cosine)")
    parser.add_argument("--opt__step_size", type=int, default=None,
                        help="Step size for lr scheduler.")
    parser.add_argument("--opt__lr_classifier_ratio", type=float, default=None,
                        help="Multiplicative factor for the classifier head "
                             "learning rate.")

    # ======================================================================
    #                              MODEL
    # ======================================================================
    parser.add_argument("--arch", type=str, default=None,
                        help="model's name.")
    parser.add_argument("--encoder_name", type=str, default=None,
                        help="Name of the backbone")
    parser.add_argument("--in_channels", type=int, default=None,
                        help="Input channels number.")
    parser.add_argument("--strict", type=str2bool, default=None,
                        help="strict mode for loading weights.")
    parser.add_argument("--encoder_weights", type=str, default=None,
                        help="Pre-trained weights.")
    parser.add_argument("--path_pre_trained", type=str, default=None,
                        help="Absolute/relative path to file of weights.")
    parser.add_argument("--support_background", type=str2bool, default=None,
                        help="use or not 1 extra plan for background cams.")
    parser.add_argument("--scale_in", type=float, default=None,
                        help="How much to scale the input.")

    parser.add_argument("--freeze_cl", type=str2bool, default=None,
                        help="whether or not to freeze the classifier.")
    parser.add_argument("--folder_pre_trained_cl", type=str, default=None,
                        help="NAME of folder containing classifier's "
                             "weights.")

    # ======================================================================
    #                    CLASSIFICATION SPATIAL POOLING
    # ======================================================================
    parser.add_argument("--method", type=str, default=None,
                        help="Name of method.")
    parser.add_argument("--spatial_pooling", type=str, default=None,
                        help="Name of spatial pooling for classification.")
    # ======================================================================
    #                        WILDCAT POOLING
    # ======================================================================

    parser.add_argument("--wc_alpha", type=float, default=None,
                        help="Alpha (classifier, wildcat)")
    parser.add_argument("--wc_kmax", type=float, default=None,
                        help="Kmax (classifier, wildcat)")
    parser.add_argument("--wc_kmin", type=float, default=None,
                        help="Kmin (classifier, wildcat)")
    parser.add_argument("--wc_dropout", type=float, default=None,
                        help="Dropout (classifier, wildcat)")
    parser.add_argument("--wc_modalities", type=int, default=None,
                        help="Number of modalities (classifier, wildcat)")

    parser.add_argument("--lse_r", type=float, default=None,
                        help="LSE r pooling.")

    # ======================================================================
    #                         EXTRA - MODE
    # ======================================================================

    parser.add_argument("--seg_mode", type=str, default=None,
                        help="Segmentation mode.")
    parser.add_argument("--task", type=str, default=None,
                        help="Type of the task.")
    parser.add_argument("--multi_label_flag", type=str2bool, default=None,
                        help="Whether the dataset is multi-label.")
    # ======================================================================
    #                         ELB
    # ======================================================================
    parser.add_argument("--elb_init_t", type=float, default=None,
                        help="Init t for elb.")
    parser.add_argument("--elb_max_t", type=float, default=None,
                        help="Max t for elb.")
    parser.add_argument("--elb_mulcoef", type=float, default=None,
                        help="Multi. coef. for elb..")

    # ======================================================================
    #                         CONSTRAINTS
    # ======================================================================
    parser.add_argument("--crf_fc", type=str2bool, default=None,
                        help="CRF over fcams flag.")
    parser.add_argument("--crf_lambda", type=float, default=None,
                        help="Lambda for crf flag.")
    parser.add_argument("--crf_sigma_rgb", type=float, default=None,
                        help="sigma rgb of crf flag.")
    parser.add_argument("--crf_sigma_xy", type=float, default=None,
                        help="sigma xy for crf flag.")
    parser.add_argument("--crf_scale", type=float, default=None,
                        help="scale factor for crf flag.")
    parser.add_argument("--crf_start_ep", type=int, default=None,
                        help="epoch start crf loss.")
    parser.add_argument("--crf_end_ep", type=int, default=None,
                        help="epoch end crf loss. use -1 for end training.")

    parser.add_argument("--entropy_fc", type=str2bool, default=None,
                        help="Entropy over fcams flag.")
    parser.add_argument("--entropy_fc_lambda", type=float, default=None,
                        help="lambda for entropy over fcams flag.")

    parser.add_argument("--max_sizepos_fc", type=str2bool, default=None,
                        help="Max size pos fcams flag.")
    parser.add_argument("--max_sizepos_fc_lambda", type=float, default=None,
                        help="lambda for max size low pos fcams flag.")
    parser.add_argument("--max_sizepos_fc_start_ep", type=int, default=None,
                        help="epoch start maxsz loss.")
    parser.add_argument("--max_sizepos_fc_end_ep", type=int, default=None,
                        help="epoch end maxsz. -1 for end training.")

    parser.add_argument("--im_rec", type=str2bool, default=None,
                        help="image reconstruction flag.")
    parser.add_argument("--im_rec_lambda", type=float, default=None,
                        help="Lambda for image reconstruction.")
    parser.add_argument("--im_rec_elb", type=str2bool, default=None,
                        help="use/not elb for image reconstruction.")

    parser.add_argument("--sl_fc", type=str2bool, default=None,
                        help="Self-learning over fcams.")
    parser.add_argument("--sl_fc_lambda", type=float, default=None,
                        help="Lambda for self-learning fcams.")
    parser.add_argument("--sl_start_ep", type=int, default=None,
                        help="Start epoch for self-learning fcams.")
    parser.add_argument("--sl_end_ep", type=int, default=None,
                        help="End epoch for self-learning fcams.")
    parser.add_argument("--sl_min", type=int, default=None,
                        help="MIN for self-learning fcams.")
    parser.add_argument("--sl_max", type=int, default=None,
                        help="MAX for self-learning fcams.")
    parser.add_argument("--sl_ksz", type=int, default=None,
                        help="Kernel size for dilation for self-learning "
                             "fcams.")
    parser.add_argument("--sl_min_p", type=float, default=None,
                        help="Percentage of pixels to be considered "
                             "background to sample from.")
    parser.add_argument("--sl_fg_erode_k", type=int, default=None,
                        help="Kernel size of erosion for foreground.")
    parser.add_argument("--sl_fg_erode_iter", type=int, default=None,
                        help="Number of time to perform erosion over "
                             "foreground.")
    parser.add_argument("--sl_min_ext", type=int, default=None,
                        help="MIN extent for self-learning fcams.")
    parser.add_argument("--sl_max_ext", type=int, default=None,
                        help="MAX extent for self-learning fcams.")
    parser.add_argument("--sl_block", type=int, default=None,
                        help="Size of the blocks for self-learning fcams.")

    parser.add_argument("--seg_ignore_idx", type=int, default=None,
                        help="Ignore index for segmentation.")

    input_parser = parser.parse_args()

    def warnit(name, vl_old, vl):
        """
        Warn that the variable with the name 'name' has changed its value
        from 'vl_old' to 'vl' through command line.
        :param name: str, name of the variable.
        :param vl_old: old value.
        :param vl: new value.
        :return:
        """
        if vl_old != vl:
            print("Changing {}: {}  -----> {}".format(name, vl_old, vl))
        else:
            print("{}: {}".format(name, vl_old))

    attributes = input_parser.__dict__.keys()

    for k in attributes:
        val_k = getattr(input_parser, k)
        if k in args.keys():
            if val_k is not None:
                warnit(k, args[k], val_k)
                args[k] = val_k
            else:
                warnit(k, args[k], args[k])

        elif k in args['model'].keys():  # try model
            if val_k is not None:
                warnit('model.{}'.format(k), args['model'][k], val_k)
                args['model'][k] = val_k
            else:
                warnit('model.{}'.format(k), args['model'][k],
                       args['model'][k])

        elif k in args['optimizer'].keys():  # try optimizer 0
            if val_k is not None:
                warnit(
                    'optimizer.{}'.format(k), args['optimizer'][k], val_k)
                args['optimizer'][k] = val_k
            else:
                warnit(
                    'optimizer.{}'.format(k), args['optimizer'][k],
                    args['optimizer'][k]
                )
        else:
            raise ValueError("Key {} was not found in args. ..."
                             "[NOT OK]".format(k))

    # add the current seed to the os env. vars. to be shared across this
    # process.
    # this seed is expected to be local for this process and all its
    # children.
    # running a parallel process will not have access to this copy not
    # modify it. Also, this variable will not appear in the system list
    # of variables. This is the expected behavior.
    # TODO: change this way of sharing the seed through os.environ. [future]
    # the doc mentions that the above depends on `putenv()` of the
    # platform.
    # https://docs.python.org/3.7/library/os.html#os.environ
    os.environ['MYSEED'] = str(args["MYSEED"])
    reproducibility.set_to_deterministic(verbose=True)

    args['outd'], args['subpath'] = outfd(Dict2Obj(args), eval=eval)
    args['outd_backup'] = args['outd']
    if is_cc():
        _tag = '{}__{}'.format(
            basename(normpath(args['outd'])), '{}_{}'.format(
                dt.datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f'),
                np.random.randint(low=0, high=10000000, size=1)[0]))
        args['outd'] = join(os.environ["SLURM_TMPDIR"], _tag)
        mkdir(args['outd'])

    cmdr = not constants.OVERRUN
    cmdr &= not eval
    if is_cc():
        cmdr &= os.path.isfile(join(args['outd_backup'], 'passed.txt'))
    else:
        cmdr &= os.path.isfile(join(args['outd'], 'passed.txt'))
    if cmdr:
        warnings.warn('EXP {} has already been done. EXITING.'.format(
            args['outd']))
        sys.exit(0)

    args['scoremap_paths'] = configure_scoremap_output_paths(Dict2Obj(args))

    if args['box_v2_metric']:
        args['multi_contour_eval'] = True
        args['multi_iou_eval'] = True
    else:
        args['multi_contour_eval'] = False
        args['multi_iou_eval'] = False

    if args['model']['freeze_cl']:
        tag = get_tag(Dict2Obj(args),
                      checkpoint_type=args['eval_checkpoint_type'])
        args['model']['folder_pre_trained_cl'] = join(
            root_dir, 'pretrained', tag)

        assert os.path.isdir(args['model']['folder_pre_trained_cl'])

    if args['task'] == constants.F_CL:
        for split in constants.SPLITS:
            tag = get_tag(Dict2Obj(args),
                          checkpoint_type=args['eval_checkpoint_type'])
            tag += '_cams_{}'.format(split)

            if is_cc():
                baseurl_sc = "{}/datasets/wsol-done-right".format(
                    os.environ["SCRATCH"])
                scratch_path = join(baseurl_sc, '{}.tar.gz'.format(tag))

                if os.path.isfile(scratch_path):
                    slurm_dir = config.get_root_wsol_dataset()
                    cmds = [
                        'cp {} {} '.format(scratch_path, slurm_dir),
                        'cd {} '.format(slurm_dir),
                        'tar -xf {}'.format('{}.tar.gz'.format(tag))
                    ]
                    cmdx = " && ".join(cmds)
                    print("Running bash-cmds: \n{}".format(
                        cmdx.replace("&& ", "\n")))
                    subprocess.run(cmdx, shell=True, check=True)

                    assert os.path.isdir(join(slurm_dir, tag))
                    args['std_cams_folder'][split] = join(slurm_dir, tag)

            else:
                path_cams = join(root_dir, constants.DATA_CAMS, tag)
                cndx = not os.path.isdir(path_cams)
                cndx &= os.path.isfile('{}.tar.gz'.format(path_cams))
                if cndx:
                    cmds_untar = [
                        'cd {} '.format(join(root_dir, constants.DATA_CAMS)),
                        'tar -xf {} '.format('{}.tar.gz'.format(tag))
                    ]
                    cmdx = " && ".join(cmds_untar)
                    print("Running bash-cmds: \n{}".format(
                        cmdx.replace("&& ", "\n")))
                    subprocess.run(cmdx, shell=True, check=True)

                if os.path.isdir(path_cams):
                    args['std_cams_folder'][split] = path_cams

    args_dict = deepcopy(args)
    args = Dict2Obj(args)
    # sanity check ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    assert args.spatial_pooling == constants.METHOD_2_POOLINGHEAD[args.method]
    assert args.model['encoder_name'] in constants.BACKBONES
    
    assert not args.multi_label_flag
    assert args.seg_mode == constants.BINARY_MODE

    if isinstance(args.resize_size, int):
        if isinstance(args.crop_size, int):
            assert args.resize_size >= args.crop_size

    # todo
    assert args.model['scale_in'] > 0.
    assert isinstance(args.model['scale_in'], float)

    if args.task == constants.STD_CL:
        assert not args.model['freeze_cl']
        assert args.model['folder_pre_trained_cl'] in [None, '', 'None']


    used_constraints = [args.sl_fc,
                        args.crf_fc,
                        args.entropy_fc,
                        args.max_sizepos_fc]

    if args.task == constants.STD_CL:
        assert not any(used_constraints)

    assert args.resize_size == constants.RESIZE_SIZE
    assert args.crop_size == constants.CROP_SIZE

    if args.task == constants.F_CL:
        assert any(used_constraints)
        assert args.model['arch'] == constants.UNETFCAM

        assert args.eval_checkpoint_type == constants.BEST

    assert args.model['arch'] in constants.ARCHS

    assert not args.im_rec

    return args, args_dict


def configure_scoremap_output_paths(args):
    scoremaps_root = join(args.outd, 'scoremaps')
    scoremaps = mch()
    for split in (constants.TRAINSET, constants.VALIDSET, constants.TESTSET):
        scoremaps[split] = join(scoremaps_root, split)
        if not os.path.isdir(scoremaps[split]):
            os.makedirs(scoremaps[split])
    return scoremaps


def outfd(args, eval=False):

    tag = [('id', args.exp_id),
           ('tsk', args.task),
           ('ds', args.dataset),
           ('mth', args.method),
           ('spooling', args.spatial_pooling),
           ('sd', args.MYSEED),
           ('ecd', args.model['encoder_name']),
           ('epx', args.max_epochs),
           ('bsz', args.batch_size),
           ('lr', args.optimizer['opt__lr']),
           ('box_v2_metric', args.box_v2_metric)
           ]

    tag = [(el[0], str(el[1])) for el in tag]
    tag = '-'.join(['_'.join(el) for el in tag])

    if args.task == constants.F_CL:
        # todo: add hyper-params.
        tag2 = []

        if args.sl_fc:
            tag2.append(("sl_fc", 'yes'))

        if args.crf_fc:
            tag2.append(("crf_fc", 'yes'))

        if args.entropy_fc:
            tag2.append(("entropy_fc", 'yes'))

        if args.max_sizepos_fc:
            tag2.append(("max_sizepos_fc", 'yes'))

        if tag2:
            tag2 = [(el[0], str(el[1])) for el in tag2]
            tag2 = '-'.join(['_'.join(el) for el in tag2])
            tag = "{}-{}".format(tag, tag2)

    parent_lv = "exps"
    if args.debug_subfolder not in ['', None, 'None']:
        parent_lv = join(parent_lv, args.debug_subfolder)

    subfd = join(args.dataset, args.model['encoder_name'], args.task,
                 args.method)
    _root_dir = root_dir
    if is_cc():
        _root_dir = join(os.environ["SCRATCH"], 'fcam')

    subpath = join(parent_lv,
                   subfd,
                   tag)
    if not eval:
        OUTD = join(_root_dir,
                    subpath
                    )
    else:
        OUTD = join(_root_dir, args.fd_exp)

    OUTD = expanduser(OUTD)

    if not os.path.exists(OUTD):
        os.makedirs(OUTD)

    return OUTD, subpath


def wrap_sys_argv_cmd(cmd: str, pre):
    splits = cmd.split(' ')
    el = splits[1:]
    pairs = ['{} {}'.format(i, j) for i, j in zip(el[::2], el[1::2])]
    pro = splits[0]
    sep = ' \\\n' + (len(pre) + len(pro) + 2) * ' '
    out = sep.join(pairs)
    return "{} {} {}".format(pre, pro, out)


def copy_code(dest,
              compress=False,
              verbose=False
              ):
    """Copy code to the exp folder for reproducibility.
    Input:
        dest: path to the destination folder (the exp folder).
        compress: bool. if true, we compress the destination folder and
        delete it.
        verbose: bool. if true, we show what is going on.
    """
    # extensions to copy.
    exts = tuple(["py", "sh", "yaml"])
    flds_files = ['.']

    for fld in flds_files:
        files = glob.iglob(os.path.join(root_dir, fld, "*"))
        subfd = join(dest, fld) if fld != "." else dest
        if not os.path.exists(subfd):
            os.makedirs(subfd, exist_ok=True)

        for file in files:
            if file.endswith(exts):
                if os.path.isfile(file):
                    shutil.copy(file, subfd)
    # cp dlib
    dirs = ["dlib", "cmds"]
    for dirx in dirs:
        cmds = [
            "cd {} && ".format(root_dir),
            "cp -r {} {} ".format(dirx, dest)
        ]
        cmd = "".join(cmds)
        if verbose:
            print("Running bash-cmds: \n{}".format(cmd.replace("&& ", "\n")))
        subprocess.run(cmd, shell=True, check=True)

    if compress:
        head = dest.split(os.sep)[-1]
        if head == '':  # dest ends with '/'
            head = dest.split(os.sep)[-2]
        cmds = [
            "cd {} && ".format(dest),
            "cd .. && ",
            "tar -cf {}.tar.gz {}  && ".format(head, head),
            "rm -rf {}".format(head)
               ]

        cmd = "".join(cmds)
        if verbose:
            print("Running bash-cmds: \n{}".format(cmd.replace("&& ", "\n")))
        subprocess.run(cmd, shell=True, check=True)


def parse_input(eval=False):
    """
    Parse the input.
    and
    initialize some modules for reproducibility.
    """
    parser = argparse.ArgumentParser()

    if not eval:
        parser.add_argument("--dataset", type=str,
                            help="Dataset name: {}.".format(constants.datasets))
        input_args, _ = parser.parse_known_args()
        args: dict = config.get_config(input_args.dataset)
        args, args_dict = get_args(args)

        log_backends = [
            ArbJSONStreamBackend(Verbosity.VERBOSE,
                                 join(args.outd, "log.json")),
            ArbTextStreamBackend(Verbosity.VERBOSE,
                                 join(args.outd, "log.txt")),
        ]

        if args.verbose:
            log_backends.append(ArbStdOutBackend(Verbosity.VERBOSE))
        DLLogger.init_arb(backends=log_backends)
        DLLogger.log(fmsg("Start time: {}".format(args.t0)))

        __split = constants.TRAINSET
        if os.path.isdir(args.std_cams_folder[__split]):
            msg = 'Will be using PRE-computed cams for split {} from ' \
                  '{}'.format(__split, args.std_cams_folder[__split])
            warnings.warn(msg)
            DLLogger.log(msg)
        else:
            msg = 'Will RE-computed cams for split {}.'.format(__split)
            if args.task == constants.F_CL:
                warnings.warn(msg)
                DLLogger.log(msg)

        outd = args.outd
        if not os.path.exists(join(outd, "code/")):
            os.makedirs(join(outd, "code/"))

        with open(join(outd, "code/config.yml"), 'w') as fyaml:
            yaml.dump(args_dict, fyaml)

        with open(join(outd, "config.yml"), 'w') as fyaml:
            yaml.dump(args_dict, fyaml)

        str_cmd = wrap_sys_argv_cmd(" ".join(sys.argv), "time python")
        with open(join(outd, "code/cmd.sh"), 'w') as frun:
            frun.write("#!/usr/bin/env bash \n")
            frun.write(str_cmd)

        copy_code(join(outd, "code/"), compress=True, verbose=False)
    else:
        parser.add_argument("--fd_exp", type=str,
                            help="relative path to the exp folder.")
        input_args, _ = parser.parse_known_args()
        _root_dir = root_dir
        if is_cc():
            _root_dir = join(os.environ["SCRATCH"], 'fcam')

        fd = join(_root_dir, input_args.fd_exp)

        # cmdx = ' cd {} && tar -xvf code.tar.gz'.format(fd)
        # print('Running: {}'.format(cmdx))
        #
        # try:
        #     subprocess.run(cmdx, shell=True, check=True)
        # except subprocess.SubprocessError as e:
        #     print("Failed to run: {}. Error: {}".format(cmdx, e))
        # files = find_files_pattern(join(fd, 'code'), '*.yaml')
        # assert len(files) == 1
        # yaml_file = files[0]

        yaml_file = join(fd, 'config.yaml')
        with open(yaml_file, 'r') as fy:
            args = yaml.load(fy)

        args, args_dict = get_args(args, eval)

    DLLogger.flush()
    return args, args_dict
