import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure.constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

from dlib.losses.jaccard import JaccardLoss
from dlib.losses.dice import DiceLoss
from dlib.losses.focal import FocalLoss
from dlib.losses.lovasz import LovaszLoss
from dlib.losses.soft_bce import SoftBCEWithLogitsLoss
from dlib.losses.soft_ce import SoftCrossEntropyLoss

from dlib.losses.core import MasterLoss
from dlib.losses.core import ClLoss
from dlib.losses.core import ImgReconstruction
from dlib.losses.core import SelfLearningFcams
from dlib.losses.core import ConRanFieldFcams
from dlib.losses.core import EntropyFcams
from dlib.losses.core import MaxSizePositiveFcams
