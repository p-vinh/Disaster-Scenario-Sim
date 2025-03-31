#!/usr/bin/env python

#####################################################################################################################################################################
# xView2                                                                                                                                                            #
# Copyright 2019 Carnegie Mellon University.                                                                                                                        #
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO    #
# WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY,          # 
# EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, # 
# TRADEMARK, OR COPYRIGHT INFRINGEMENT.                                                                                                                             #
# Released under a MIT (SEI)-style license, please see LICENSE.md or contact permission@sei.cmu.edu for full terms.                                                 #
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use  #
# and distribution.                                                                                                                                                 #
# This Software includes and/or makes use of the following Third-Party Software subject to its own license:                                                         #
# 1. SpaceNet (https://github.com/motokimura/spacenet_building_detection/blob/master/LICENSE) Copyright 2017 Motoki Kimura.                                         #
# DM19-0988                                                                                                                                                         #
#####################################################################################################################################################################

# segmentation_keras.py

import numpy as np
import cv2
import math
import tensorflow as tf
from tensorflow.keras.models import load_model

class SegmentationModel:
    def __init__(self, model_path, mean):
        self.model = load_model(model_path)
        self.mean = mean[np.newaxis, np.newaxis, :]

    def apply_segmentation(self, image):
        image_in, crop = self.__preprocess(image)
        score = self.model(image_in, training=False).numpy()[0]
        top, left, bottom, right = crop
        score = score[:, top:bottom, left:right]
        return score

    def apply_segmentation_to_mosaic(self, mosaic, grid_px=800, tile_overlap_px=200):
        h, w, _ = mosaic.shape
        assert ((grid_px + tile_overlap_px * 2) % 16 == 0), "(grid_px + tile_overlap_px * 2) must be divisible by 16"

        pad_y1 = tile_overlap_px
        pad_x1 = tile_overlap_px
        n_y = int(h / grid_px)
        n_x = int(w / grid_px)

        pad_y2 = n_y * grid_px + 2 * tile_overlap_px - h - pad_y1
        pad_x2 = n_x * grid_px + 2 * tile_overlap_px - w - pad_x1

        mosaic_padded = np.pad(mosaic, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)), 'symmetric')
        H, W, _ = mosaic_padded.shape
        score_padded = np.zeros((self.model.output_shape[1], H, W), dtype=np.float32)

        for yi in range(n_y):
            for xi in range(n_x):
                top = yi * grid_px
                left = xi * grid_px
                bottom = top + grid_px + 2 * tile_overlap_px
                right = left + grid_px + 2 * tile_overlap_px

                tile = mosaic_padded[top:bottom, left:right]
                score_tile = self.apply_segmentation(tile)
                score_padded[:, top:bottom, left:right] = score_tile

        return score_padded[:, pad_y1:-pad_y2, pad_x1:-pad_x2]

    def __preprocess(self, image):
        h, w, _ = image.shape
        h_padded = int(math.ceil(h / 16.0) * 16)
        w_padded = int(math.ceil(w / 16.0) * 16)

        pad_y1 = (h_padded - h) // 2
        pad_x1 = (w_padded - w) // 2
        pad_y2 = h_padded - h - pad_y1
        pad_x2 = w_padded - w - pad_x1

        image_padded = np.pad(image, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)), 'symmetric')
        image_in = (image_padded - self.mean) / 255.0
        image_in = image_in[np.newaxis].astype(np.float32)

        return image_in, (pad_y1, pad_x1, pad_y1 + h, pad_x1 + w)
