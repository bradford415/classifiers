from typing import Optional, Union

from .simmim import build_swin_simmim
from .swin import build_swin

classifier_map = {
    "swin": build_swin,
}


def create_classifier(
    classifier_name: str,
    classifier_args: dict[str, any],
    num_classes: int,
    image_size: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
):
    """Initialize the desired object detection model

    Args:
        classifier_name: name of the classifier architecture to initialize
        classifier_args: a dictionary of the parameters specific to the detector class
        num_classes: number of classes in the dataset ontology; number of output neurons the
                     final linear layer

    """
    if "swin" in classifier_name:
        model = _create_swin(num_classes, image_size, classifier_args, checkpoint_path)
    else:
        raise ValueError(f"detctor: {classifier_name} not recognized")

    return model


def create_simmim_model(
    backbone_name: str,
    backbone_args: dict[str, any],
    image_size: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
):
    """Initialize the desired object detection model

    Args:
        classifier_name: name of the classifier architecture to initialize
        classifier_args: a dictionary of the parameters specific to the detector class
        image_size: the image size of the input to the model


    """
    if "swin" in backbone_name:
        encoder_stride = 32
        model = build_swin_simmim(image_size, encoder_stride, backbone_args)
    else:
        raise ValueError(f"backbone: {backbone_name} not recognized")

    return model


def _create_swin(
    num_classes: int,
    img_size: Union[int, tuple],
    classifier_args: dict[str, any],
    checkpoint_path: Optional[str],
):
    """Create the Swin Transformer

    Args:
        classifier_name: TODO
        checkpoint_path: the path to a checkpoint file (.pth) to finetune from or resume training from;
                         e.g., this could be a pretrained simmim model or Swin classifier you want to finish
                         training
    """
    model = classifier_map["swin"](
        num_classes=num_classes,
        img_size=img_size,
        swin_params=classifier_args,
    )
    return model
