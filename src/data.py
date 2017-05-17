"""
Module containing the data loading functionality.
"""

import collections
import csv
import os

import settings
import utils

logger = settings.logger.getChild('data')

class Loader:

    def __init__(self):
        self.train_counts = self._load_train_counts()
        self.train_coordinates = self._load_train_coordinates()
        self.train_mismatched = self._load_train_mismatched()

    def _load_train_counts(self):
        """
        Load the counts CSV for the training dataset.
        Turn it into a dictionary of dictionaries of counts.

        {image_id: {
                adult_males: n
                subadult_males: n
                adult_females: n
                juveniles: n
                pups: n
            }
        }
        :return: A dictionary of dictionaries of counts
        """
        logger.debug('Loading train image counts')
        with open(settings.TRAIN_COUNTS_CSV, 'r') as file:
            d = {row['train_id']: utils.remove_key_from_dict(row, 'train_id') for row in csv.DictReader(file)}
        return d

    def _load_train_coordinates(self):
        """
        Load the coordinates CSV for the training dataset.
        Turn it into a dictionary of lists of coordinates and labels.
        
        {image_id: [
                {
                    x_coord: n
                    y_coord: n
                    category: label
                }
            ]
        }

        :return: A dictionary of lists of coordinates and categories.
        """
        logger.debug('Loading train image coordinates')
        d = collections.defaultdict(list)
        with open(settings.TRAIN_COORDINATES_CSV, 'r') as file:
            [d[utils.get_file_name_part(row['filename'])].append(utils.remove_key_from_dict(row, 'id', 'filename')) for row in csv.DictReader(file)]
        return dict(d)

    def _load_train_mismatched(self):
        """
        Load the mismatched image IDs CSV for the training dataset.
        Turn it into a dictionary of booleans (all true) for fast querying via "in".

        :return: A dictionary containing entries for images that are mismatched.
        """
        logger.debug('Loading train image mismatch labels')
        with open(settings.TRAIN_MISMATACHED_CSV, 'r') as file:
            d= {row['train_id']: True for row in csv.DictReader(file)}
        return d

    def get_train_counts(self):
        return self.train_counts

    def get_train_coordinates(self):
        return self.train_coordinates

    def get_train_mismatched(self):
        return self.train_mismatched
