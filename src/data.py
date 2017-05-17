"""
Module containing the data loading functionality.
"""

import collections
import csv
import glob
import os
import threading

import sklearn.model_selection

import settings
import utils

logger = settings.logger.getChild('data')

class Loader:

    def __init__(self):
        self.train_original_counts = self._load_train_original_counts()
        self.train_original_coordinates = self._load_train_original_coordinates()
        self.train_original_mismatched = self._load_train_original_mismatched()

    def _load_train_original_counts(self):
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

    def _load_train_original_coordinates(self):
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

    def _load_train_original_mismatched(self):
        """
        Load the mismatched image IDs CSV for the training dataset.
        Turn it into a dictionary of booleans (all true) for fast querying via "in".

        :return: A dictionary containing entries for images that are mismatched.
        """
        logger.debug('Loading train image mismatch labels')
        with open(settings.TRAIN_MISMATACHED_CSV, 'r') as file:
            d= {row['train_id']: True for row in csv.DictReader(file)}
        return d

    def get_train_original_counts(self):
        return self.train_original_counts

    def get_train_original_coordinates(self):
        return self.train_original_coordinates

    def get_train_original_mismatched(self):
        return self.train_original_mismatched

    def load_original_images(self, dataset = "train"):
        """
        Load the data
        """
        
        images = []

        if dataset == "train":

            # Get all train original images
            filenames = sorted(glob.glob(os.path.join(settings.TRAIN_ORIGINAL_IMAGES_DIR, "*.jpg")))
            for filename in filenames:
                name = utils.get_file_name_part(filename)

                if name in self.train_original_mismatched:
                    # Skip images marked as mismatched
                    continue

                meta = {
                    'filename': name,
                    'coordinates': self.train_original_coordinates[name] if name in self.train_original_coordinates else [],
                    'counts': self.train_original_counts
                }

                images.append({'x': (lambda filename: lambda: self.load(filename))(filename),
                               'm': meta})

        elif dataset == "test_st1":

            # Get all test original images
            filenames = sorted(glob.glob(os.path.join(settings.TEST_ORIGINAL_IMAGES_DIR, "*.jpg")))
            for filename in filenames:
                name = self.get_file_name_part(filename)

                if name in self.train_original_mismatched:
                    # Skip images marked as mismatched
                    continue

                meta = {
                    'filename': name
                }

                images.append({'x': (lambda filename: lambda: self.load(filename))(filename),
                               'm': meta})

        return images

    def train_val_split(self, data, split_ratio = 0.7):
        """
        Split a dataset into a training and validation set.

        :param data: The list of data to split
        :param split_ratio: The ratio to use, e.g. 0.7 means 70% of the data will be used for training
        :return: A dictionary of training and validation data {'train': ..., 'validate': ...}
        """
        data_train, data_val = sklearn.model_selection.train_test_split(data, train_size = split_ratio, random_state = 42)
        return {'train': data_train, 'validate': data_val}


    def load(self, filename):
        """
        Load an image into a scipy ndarray

        :param filename: The name of the file to load
        :return: The image as an ndarray
        """
        return scipy.misc.imread(filename).astype("float32")

class Iterator(object):
    # See: https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

class DataIterator(Iterator):
    def __init__(self, images):
        super(DataIterator, self).__init__(len(images))
