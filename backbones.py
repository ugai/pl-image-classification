from enum import Enum

import torchvision.models as TVM
from torch import nn
from torchvision.models._api import Weights


class BackboneType(str, Enum):
    """Type of pretrained model"""

    ResNet50 = "resnet50"
    ResNet101 = "resnet101"
    ResNet152 = "resnet152"

    ResNext50 = "resnext50"
    ResNext101_32 = "resnext101_32"
    ResNext101_64 = "resnext101_64"

    EfficientNetV2Small = "efficientnetv2s"
    EfficientNetV2Medium = "efficientnetv2m"
    EfficientNetV2Large = "efficientnetv2l"

    VitH14 = "vith14"

    MaxVitT = "maxvitt"

    SwinV2Tiny = "swinv2t"
    SwinV2Small = "swinv2s"
    SwinV2Base = "swinv2b"


def get_backbone_model_and_weights(
    backbone_type: BackboneType,
) -> tuple[nn.Module, Weights]:
    """Get pretraind model and weights"""
    match backbone_type:
        # ResNet
        # https://pytorch.org/vision/stable/models/resnet.html
        case BackboneType.ResNet50:
            weights: Weights = TVM.ResNet50_Weights.DEFAULT  # type: ignore
            model: nn.Module = TVM.resnet50(weights=weights)
        case BackboneType.ResNet101:
            weights: Weights = TVM.ResNet101_Weights.DEFAULT  # type: ignore
            model: nn.Module = TVM.resnet101(weights=weights)
        case BackboneType.ResNet152:
            weights: Weights = TVM.ResNet152_Weights.DEFAULT  # type: ignore
            model: nn.Module = TVM.resnet152(weights=weights)

        # ResNext
        # https://pytorch.org/vision/stable/models/resnext.html
        case BackboneType.ResNext50:
            weights: Weights = TVM.ResNeXt50_32X4D_Weights.DEFAULT  # type: ignore
            model: nn.Module = TVM.resnext50_32x4d(weights=weights)
        case BackboneType.ResNext101_32:
            weights: Weights = TVM.ResNeXt101_32X8D_Weights.DEFAULT  # type: ignore
            model: nn.Module = TVM.resnext101_32x8d(weights=weights)
        case BackboneType.ResNext101_64:
            weights: Weights = TVM.ResNeXt101_64X4D_Weights.DEFAULT  # type: ignore
            model: nn.Module = TVM.resnext101_64x4d(weights=weights)

        # EfficientNetV2
        # https://pytorch.org/vision/stable/models/efficientnetv2.html
        case backbone_type.EfficientNetV2Small:
            weights: Weights = TVM.EfficientNet_V2_S_Weights.DEFAULT  # type: ignore
            model: nn.Module = TVM.efficientnet_v2_s(weights=weights)
        case backbone_type.EfficientNetV2Medium:
            weights: Weights = TVM.EfficientNet_V2_M_Weights.DEFAULT  # type: ignore
            model: nn.Module = TVM.efficientnet_v2_m(weights=weights)
        case backbone_type.EfficientNetV2Large:
            weights: Weights = TVM.EfficientNet_V2_L_Weights.DEFAULT  # type: ignore
            model: nn.Module = TVM.efficientnet_v2_l(weights=weights)

        # TODO: fix error
        # VisionTransformer
        # https://pytorch.org/vision/stable/models/vision_transformer.html
        case backbone_type.VitH14:
            weights: Weights = TVM.ViT_H_14_Weights.DEFAULT  # type: ignore
            model: nn.Module = TVM.vit_h_14(weights=weights)

        # TODO: fix error
        # MaxVit
        # https://pytorch.org/vision/stable/models/maxvit.html
        case backbone_type.MaxVitT:
            weights: Weights = TVM.MaxVit_T_Weights.DEFAULT  # type: ignore
            model: nn.Module = TVM.maxvit_t(weights=weights)

        # SwinTransformer V2
        # https://pytorch.org/vision/stable/models/swin_transformer.html
        case BackboneType.SwinV2Tiny:
            weights: Weights = TVM.Swin_V2_T_Weights.DEFAULT  # type: ignore
            model: nn.Module = TVM.swin_v2_t(weights=weights)
        case BackboneType.SwinV2Small:
            weights: Weights = TVM.Swin_V2_S_Weights.DEFAULT  # type: ignore
            model: nn.Module = TVM.swin_v2_s(weights=weights)
        case BackboneType.SwinV2Base:
            weights: Weights = TVM.Swin_V2_B_Weights.DEFAULT  # type: ignore
            model: nn.Module = TVM.swin_v2_b(weights=weights)

        case _:
            raise NotImplementedError()

    return model, weights
