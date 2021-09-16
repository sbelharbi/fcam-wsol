import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from typing import Optional
from dlib.deeplabv3.decoder import DeepLabV3Decoder, DeepLabV3PlusDecoder
from dlib.base import SegmentationModel, SegmentationHead, ClassificationHead
from dlib.encoders import get_encoder


class DeepLabV3(SegmentationModel):
    """DeepLabV3_ implementation from "Rethinking Atrous Convolution for
    Semantic Image Segmentation"

    Args:
        encoder_name: Name of the classification model that will be used as an
        encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5].
        Each stage generate features
            two times smaller in spatial dimensions than previous one
            (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth
            1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization),
        **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for
            each encoder_name)
        decoder_channels: A number of convolution filters in ASPP module.
            Default is 256
        in_channels: A number of input channels for the model, default is 3
            (RGB images)
        classes: A number of classes for output mask (or you can think as a
            number of channels of output mask)
        activation: An activation function to apply after the final convolution
            layer.
            Available options are **"sigmoid"**, **"softmax"**,
            **"logsoftmax"**, **"tanh"**, **"identity"**,
            **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 8 to preserve
            input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output
            (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default).
            Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply
                "sigmoid"/"softmax" (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV3**

    .. _DeeplabV3:
        https://arxiv.org/abs/1706.05587

    """

    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            decoder_channels: int = 256,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 8,
            aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.encoder.make_dilated(
            stage_list=[4, 5],
            dilation_list=[2, 4]
        )

        self.decoder = DeepLabV3Decoder(
            in_channels=self.encoder.out_channels[-1],
            out_channels=decoder_channels,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None


class DeepLabV3Plus(SegmentationModel):
    """DeepLabV3+ implementation from "Encoder-Decoder with Atrous Separable
    Convolution for Semantic Image Segmentation"
    
    Args:
        encoder_name: Name of the classification model that will be used as an
            encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5].
            Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for
            depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W),
            (N, C, H // 2, W // 2)] and so on). Default is 5
        encoder_weights: One of **None** (random initialization),
            **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for
            each encoder_name)
        encoder_output_stride: Downsampling factor for last encoder features
            (see original paper for explanation)
        decoder_atrous_rates: Dilation rates for ASPP module (should be a
            tuple of 3 integer values)
        decoder_channels: A number of convolution filters in ASPP module.
            Default is 256
        in_channels: A number of input channels for the model, default is 3
            (RGB images)
        classes: A number of classes for output mask (or you can think as a
            number of channels of output mask)
        activation: An activation function to apply after the final convolution
            layer.
            Available options are **"sigmoid"**, **"softmax"**,
            **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and
            **None**. Default is **None**
        upsampling: Final upsampling factor. Default is 4 to preserve
            input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output
            (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default).
            Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply
                "sigmoid"/"softmax" (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV3Plus**
    
    Reference:
        https://arxiv.org/abs/1802.02611v3

    """
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            encoder_output_stride: int = 16,
            decoder_channels: int = 256,
            decoder_atrous_rates: tuple = (12, 24, 36),
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 4,
            aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        if encoder_output_stride == 8:
            self.encoder.make_dilated(
                stage_list=[4, 5],
                dilation_list=[2, 4]
            )

        elif encoder_output_stride == 16:
            self.encoder.make_dilated(
                stage_list=[5],
                dilation_list=[2]
            )
        else:
            raise ValueError(
                "Encoder output stride should be 8 or 16, "
                "got {}".format(encoder_output_stride)
            )

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None


if __name__ == "__main__":
    import torch
    import dlib

    from dlib.utils.shared import announce_msg
    from dlib.utils.reproducibility import set_seed

    set_seed(seed=0)
    cuda = "1"
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    encoders = dlib.encoders.get_encoder_names()
    in_channels = 3
    sample = torch.rand((2, in_channels, 41, 416)).to(DEVICE)

    for encoder_name in encoders:
        announce_msg("Testing backbone {}".format(encoder_name))

        print("Segmentation only")
        model = DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=4,
        ).to(DEVICE)
        out = model(sample)
        print(sample.shape, out.shape)

        # segmentation + classification
        print("Segmentation  + classification")
        model = DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=4,
            aux_params=dict(classes=3)
        ).to(DEVICE)
        masks, labels = model(sample)
        print(sample.shape, masks.shape, labels.shape)
        sys.exit()