"""
Module containing the project settings.
"""

import logging
import os
from time import strftime

# Directory settings

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

## Input directories

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test_st1")


TRAIN_ORIGINAL_IMAGES_DIR     = os.path.join(TRAIN_DIR, "original")
TRAIN_DOTTED_IMAGES_DIR       = os.path.join(TRAIN_DIR, "dotted")
TRAIN_LABELS_DIR              = os.path.join(TRAIN_DIR, "labels")
CROPS_INPUT_DIR               = os.path.join(TRAIN_DIR, "crops")
OVERLAP_MASKS_DIR             = os.path.join(TRAIN_DIR, "overlap_masks")
TRAIN_FEATURES_DIR            = os.path.join(TRAIN_DIR, "features")
TRAIN_HEATMAP_DIR             = os.path.join(CROPS_INPUT_DIR, "heatmaps")
REGION_CROPS_DIR              = os.path.join(CROPS_INPUT_DIR, "region_crops")
SEA_LION_CROPS_DIR            = os.path.join(CROPS_INPUT_DIR, "sea_lion_crops")
DENSITY_MAP_FEATURE_CROPS_DIR = os.path.join(CROPS_INPUT_DIR, "density_map_feature_crops")

TRAIN_COORDINATES_CSV         = os.path.join(TRAIN_LABELS_DIR, "coordinates.csv")
TRAIN_COUNTS_CSV              = os.path.join(TRAIN_LABELS_DIR, "counts.csv")
TRAIN_MISMATCHED_CSV          = os.path.join(TRAIN_LABELS_DIR, "MismatchedTrainImages.txt")


TEST_ORIGINAL_IMAGES_DIR      = os.path.join(TEST_DIR, "original")
TEST_HEATMAP_DIR              = os.path.join(TEST_DIR, "heatmaps")
TEST_FEATURES_DIR             = os.path.join(TEST_DIR, "features")

## Imagenet metadata

IMAGENET_DIR = os.path.join(DATA_DIR, "imagenet")
IMAGENET_CLSLOC_PATH = os.path.join(IMAGENET_DIR, "meta_clsloc.mat")

## Output directories 

WEIGHTS_DIR = os.path.join(DATA_DIR, "weights")#, strftime("%Y%m%dT%H%M%S"))
CROPS_OUTPUT_DIR = os.path.join(OUTPUT_DIR,"crops", strftime("%Y%m%dT%H%M%S"))

TENSORBOARD_LOGS_DIR = os.path.join(OUTPUT_DIR, 'tb_logs')
SUBMISSIONS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "submissions")

OBMS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'obms')
CNN_COORDINATES_DIR = os.path.join(OUTPUT_DIR, 'cnn_coordinates')


# Data augmentation settings
TRANSFORMATION_RESIZE_TO = (2000,2000,3)
AUGMENTATION_ROTATION_RANGE = 0,
AUGMENTATION_SHEAR_RANGE = None,#0.1
AUGMENTATION_ZOOM_RANGE = [0.2,2.0]
AUGMENTATION_WIDTH_SHIFT_RANGE = 0,#0.1
AUGMENTATION_HEIGHT_SHIFT_RANGE = 0,#0.1
AUGMENTATION_HORIZONTAL_FLIP = True,
AUGMENTATION_VERTICAL_FLIP = True,
AUGMENTATION_CHANNEL_SHIFT_RANGE = 8.0
AUGMENTATION_BLUR_RANGE = [0., 1]


# Problem-specific settings

## Heatmap settings
REGION_HEATMAP_NETWORK_WEIGHT_NAME = "region-blackout-resnet-lay107-ep006-tloss0.0064-vloss0.0099.hdf5"
INDIVIDUAL_HEATMAP_NETWORK_WEIGHT_NAME = "indiv-resnet-lay102-ep004-tloss0.0311-vloss0.0419.hdf5"

# Logging
## create logger
logger = logging.getLogger('noaa')
logger.setLevel(logging.DEBUG)

## create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

## create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

## add formatter to ch
ch.setFormatter(formatter)

## add ch to logger
logger.addHandler(ch)

def get_timestamp():
    return strftime("%Y%m%dT%H%M%S")
