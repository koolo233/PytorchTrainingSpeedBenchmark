import torchvision
from torchvision import models


def torchvision_model_dict():
    basic_model_dict = {
        "AlexNet": models.alexnet,
        "VGG11": models.vgg11,
        "VGG13": models.vgg13,
        "VGG16": models.vgg16,
        "VGG19": models.vgg19,
        "VGG11BN": models.vgg11_bn,
        "VGG13BN": models.vgg13_bn,
        "VGG16BN": models.vgg16_bn,
        "VGG19BN": models.vgg19_bn,
        "ResNet18": models.resnet18,
        "ResNet34": models.resnet34,
        "ResNet50": models.resnet50,
        "ResNet101": models.resnet101,
        "ResNet152": models.resnet152,
        "SqueezeNet1_0": models.squeezenet1_0,
        "SqueezeNet1_1": models.squeezenet1_1,
        "DenseNet121": models.densenet121,
        "DenseNet161": models.densenet161,
        "DenseNet169": models.densenet169,
        "DenseNet201": models.densenet201,
        "InceptionV3": models.inception_v3,
        "GoogleNet": models.googlenet,
        "ShuffleNetV2": models.shufflenet_v2_x1_0,
        "MobileNetV2": models.mobilenet_v2,
        "ResNeXt50": models.resnext50_32x4d,
        "ResNeXt101_32": models.resnext101_32x8d,
        "WideResNet50": models.wide_resnet50_2,
        "WideResNet101": models.wide_resnet101_2,
        "MNASNet": models.mnasnet1_0
    }

    version_str_list = torchvision.__version__.split(".")[:2]

    if version_str_list[0] == "0":
        min_version = eval(version_str_list[1])
        model_dict = basic_model_dict
        if min_version < 8:
            raise RuntimeWarning(f"Version: {torchvision.__version__} has not been fully tested, "
                                 f"which may lead to unexpected ERRORs")
        elif min_version == 8:
            pass
        else:
            append_model_dict = {
                "MobileNetV3_l": models.mobilenet_v3_large,
                "MobileNetV3_s": models.mobilenet_v3_small
            }
            if min_version > 10:
                # TODO regnet
                extend_dict = {
                    "EfficientNet_B0": models.efficientnet_b0,
                    "EfficientNet_B1": models.efficientnet_b1,
                    "EfficientNet_B2": models.efficientnet_b2,
                    "EfficientNet_B3": models.efficientnet_b3,
                    "EfficientNet_B4": models.efficientnet_b4,
                    "EfficientNet_B5": models.efficientnet_b5,
                    "EfficientNet_B6": models.efficientnet_b6,
                    "EfficientNet_B7": models.efficientnet_b7,
                }
                for key, value in extend_dict.items():
                    append_model_dict[key] = value

                if min_version > 11:
                    extend_dict = {
                        "VIT_b_32": models.vit_b_32,
                        "VIT_b_16": models.vit_b_16,
                        "VIT_l_16": models.vit_l_16,
                        "VIT_l_32": models.vit_l_32,
                        "ConvNeXt_t": models.convnext_tiny,
                        "ConvNeXt_s": models.convnext_small,
                        "ConvNeXt_b": models.convnext_base,
                        "ConvNeXt_l": models.convnext_large
                    }
                    for key, value in extend_dict.items():
                        append_model_dict[key] = value

                    if min_version > 12:
                        extend_dict = {
                            "EfficientNetV2_s": models.efficientnet_v2_s,
                            "EfficientNetV2_m": models.efficientnet_v2_m,
                            "EfficientNetV2_l": models.efficientnet_v2_l,
                            "ResNeXt101_64": models.resnext101_64x4d,
                            "Swin_s": models.swin_s,
                            "Swin_b": models.swin_b,
                            "Swin_t": models.swin_t,
                            "VIT_h_14": models.vit_h_14
                        }
                        for key, value in extend_dict.items():
                            append_model_dict[key] = value

                        if min_version > 13:
                            extend_dict = {
                                "MaxVIT_t": models.maxvit_t,
                                "SwinV2_s": models.swin_v2_s,
                                "SwinV2_b": models.swin_v2_b,
                                "SwinV2_t": models.swin_v2_t,
                            }
                        for key, value in extend_dict.items():
                            append_model_dict[key] = value

            for key, value in append_model_dict.items():
                model_dict[key] = value
    else:
        raise ValueError(f"This version is not supported: {torchvision.__version__}")

    return model_dict, version_str_list
