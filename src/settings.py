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
TRAIN_COORDINATES_CSV         = os.path.join(TRAIN_LABELS_DIR, "coordinates.csv")
TRAIN_COUNTS_CSV              = os.path.join(TRAIN_LABELS_DIR, "counts.csv")
TRAIN_MISMATACHED_CSV         = os.path.join(TRAIN_LABELS_DIR, "mismatched.csv")

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
