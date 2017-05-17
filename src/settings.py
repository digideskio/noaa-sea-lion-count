"""
Module containing the project settings.
"""

import os
from time import strftime

# Directory settings

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

## Input directories

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")


WEIGHTS_DIR = os.path.join(DATA_DIR, "weights")
TENSORBOARD_LOGS_DIR = os.path.join(DATA_DIR, 'tb_logs')

## Output directories

CROPS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "crops", strftime("%Y%m%dT%H%M%S"))
WEIGHTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "weights", strftime("%Y%m%dT%H%M%S"))

## Imagenet metadata

IMAGENET_DIR = os.path.join(DATA_DIR, "imagenet")
IMAGENET_CLSLOC_PATH = os.path.join(IMAGENET_DIR, "meta_clsloc.mat")

# Problem-specific settings

## Heatmap settings
HEATMAP_NETWORK_WEIGHT_NAME = ""
