"""
A small pytorch based SSD architecture.

Based on a design in https://github.com/pierluigiferrari/ssd_keras/tree/master.

For the original SSD architecture see, Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.-Y., & Berg, A. C. (2016). SSD: Single Shot MultiBox Detector. In Computer Vision – ECCV 2016 (pp. 21–37). doi:10.1007/978-3-319-46448-0_2
"""

from torch import Module
from torch import nn
import numpy as np


class SSD7(Module):
    """_summary_

    Args:
        Module (_type_): _description_
    """

    def identity_layer(self, tensor):
        return tensor

    def input_mean_normalization(self, tensor):
        return tensor - np.array(self.subtract_mean)

    def input_stddev_normalization(self, tensor):
        return tensor / np.array(self.divide_by_stddev)

    def __init__(
        self,
        image_size,
        n_classes,
        l2_regularization=0.0,
        min_scale=0.1,
        max_scale=0.9,
        scales=None,
        aspect_ratios_global=[0.5, 1.0, 2.0],
        aspect_ratios_per_layer=None,
        two_boxes_for_ar1=True,
        steps=None,
        offsets=None,
        clip_boxes=False,
        variances=[1.0, 1.0, 1.0, 1.0],
        coords="centroids",
        normalize_coords=False,
        subtract_mean=None,
        divide_by_stddev=None,
        swap_channels=False,
        confidence_thresh=0.01,
        iou_threshold=0.45,
        top_k=200,
        nms_max_output_size=400,
        return_predictor_sizes=False,
    ):
        self.subtract_mean = subtract_mean
        self.divide_by_stddev = divide_by_stddev

        n_predictor_layers = 4
        n_classes += 1
        l2_reg = l2_regularization
        img_height, img_width, img_channels = image_size

        ############################################################################
        # Get a few exceptions out of the way.
        ############################################################################

        if aspect_ratios_global is None and aspect_ratios_per_layer is None:
            raise ValueError(
                "`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified."
            )
        if aspect_ratios_per_layer:
            if len(aspect_ratios_per_layer) != n_predictor_layers:
                raise ValueError(
                    "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(
                        n_predictor_layers, len(aspect_ratios_per_layer)
                    )
                )

        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError(
                "Either `min_scale` and `max_scale` or `scales` need to be specified."
            )
        if scales:
            if len(scales) != n_predictor_layers + 1:
                raise ValueError(
                    "It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                        n_predictor_layers + 1, len(scales)
                    )
                )
        else:  # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
            scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

        if (
            len(variances) != 4
        ):  # We need one variance value for each of the four box coordinates
            raise ValueError(
                "4 variance values must be pased, but {} values were received.".format(
                    len(variances)
                )
            )
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError(
                "All variances must be >0, but the variances given are {}".format(
                    variances
                )
            )

        if (not (steps is None)) and (len(steps) != n_predictor_layers):
            raise ValueError(
                "You must provide at least one step value per predictor layer."
            )

        if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
            raise ValueError(
                "You must provide at least one offset value per predictor layer."
            )

    def forward():
        pass
