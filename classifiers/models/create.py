from typing import Optional

from .swin import SwinTransformer
from .yolov3 import Yolov3
from .yolov4 import Yolov4

detectors_map = {
    "yolov3": Yolov3,
    "yolov4": Yolov4,
    "dino": build_dino,
}


def create_classifier(
    classifier_name: str,
    classifier_args: dict[str, any],
    num_classes: int,
):
    """Initialize the desired object detection model

    Args:
        classifier_name: name of the classifier architecture to initialize
        classifier_args: a dictionary of the parameters specific to the detector class
        num_classes: number of classes in the dataset ontology; number of output neurons the
                     final linear layer

    """

    if classifier_name == "swin":
        model = _create_swin(classifier_name, num_classes, classifier_args)
    else:
        raise ValueError(f"detctor: {detector_name} not recognized")

    return model


def _create_dino(detector_name: str, num_classes: int, detector_args: dict[str, any]):
    """Create the dino detector, loss function, and postprocessor

    TODO: consider intializing the criterion/postprocessor separate from the model

    Args:
        detector_name: now of the object detection model
        num_classes: the max_obj_id + 1 (background); for coco this should be 91
        detector_args: a dictionary of parameters specific to the build_dino() function;
                       see models.dino.build_dino() docstring for these parameters
    """
    model = detectors_map[detector_name](
        num_classes=num_classes,
        backbone_args=detector_args["backbone"],
        dino_args=detector_args["detector"],
        aux_loss=detector_args["aux_loss"],
    )
    return model


def _create_yolov3(
    detector_name: str, num_classes: str, detector_args: dict[str, any], anchors
):
    """TODO

    Args:
        detector_name: the name of the object detector to use
        num_classes: the number of classes in the dataset; for coco this should be 80 and
                     these mapping should be contiguous
    """

    anchors, num_anchors = initialize_anchors(anchors)

    # extract backbone parameters if they're supplied; some backbones do not take parameters
    backbone_name = detector_args["backbone_name"]
    backbone_params = detector_args.get("backbone_params", {})

    # Initalize the detector backbone; typically some feature extractor
    backbone = backbone_map[backbone_name](**backbone_params)

    # detector args
    model_components = {
        "backbone": backbone,
        "num_classes": num_classes,
        "anchors": anchors,
    }
    # Initialize detection model and transfer to GPU
    model = detectors_map[detector_name](**model_components)
