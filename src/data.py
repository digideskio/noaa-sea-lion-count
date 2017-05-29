"""
Module containing the data loading functionality.
"""

import abc
import collections
import csv
import glob
import os
import threading
from ast import literal_eval

import scipy
import scipy.misc
import numpy as np
import sklearn.model_selection
import pandas as pd

from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input

import settings
import utils
import random

logger = settings.logger.getChild('data')

class Loader:

    def __init__(self):
        self.train_original_counts = self._load_train_original_counts()
        self.train_original_coordinates = self._load_train_original_coordinates()
        self.train_original_mismatched = self._load_train_original_mismatched()
        #self.herd_crops_coordinates = self._load_herd_crops_coordinates()
        
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
        
    def _load_herd_crops_coordinates(self):
        """
        UNUSED: since now we are cropping herds offline
        Load the herd crops CSV file into a pandas DataFrame.
        column names: train_id,y_coord,x_coord,is_herd
        """
        logger.debug('Loading herd crop coordinates')
        df = pd.read_csv(settings.HERD_CROPS_COORDINATES_CSV)
        return df
        
    def add_image_size_to_train_counts(self):
        '''
        Adds a tuple indicating image size to the file counts.csv
        '''
        logger.debug('Adding image sizes to train image counts file')
        df = pd.read_csv(os.path.join(settings.TRAIN_LABELS_DIR,'counts.csv'))
        
        def image_size_reader(train_id):
            filename = str(train_id)+ '.jpg'
            image = self.load(os.path.join(settings.TRAIN_ORIGINAL_IMAGES_DIR,filename))
            size = image.shape[:2]
            return size 
            
        df['size'] = df['train_id'].map(image_size_reader)
        
        #Overwrite counts.csv
        df.to_csv(os.path.join(settings.TRAIN_LABELS_DIR,'counts.csv'),index = False)
        #Update attribute
        self.train_original_counts = self._load_train_original_counts()
        
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
            [d[utils.get_file_name_part(row['filename'])].append(utils.remove_key_from_dict(row, '', 'id', 'filename')) for row in csv.DictReader(file)]
        return dict(d)

    def _load_train_original_mismatched(self):
        """
        Load the mismatched image IDs CSV for the training dataset.
        Turn it into a dictionary of booleans (all true) for fast querying via "in".

        :return: A dictionary containing entries for images that are mismatched.
        """
        logger.debug('Loading train image mismatch labels')
        with open(settings.TRAIN_MISMATCHED_CSV, 'r') as file:
            d = {row['train_id']: True for row in csv.DictReader(file)}
        return d

    def get_train_original_counts(self):
        return self.train_original_counts

    def get_train_original_coordinates(self):
        return self.train_original_coordinates

    def get_train_original_mismatched(self):
        return self.train_original_mismatched
        
    def load_crop_images(self, data_type):
        """
        Load precropped data
        """
        
        assert data_type in ['sea_lion_crops', 'region_crops']
        if data_type == 'region_crops':
            crops_dir = settings.REGION_CROPS_DIR
        else:
            crops_dir = settings.SEA_LION_CROPS_DIR
            assert False
        logger.debug('Loading train set '+data_type+' images')
        images = []
        # Get all train original images
        filenames = sorted(glob.glob(os.path.join(crops_dir, "*.jpg")))
        for filename in filenames:
            name = utils.get_file_name_part(filename)
            image_name = name.split('_')[1].split('d')[1]+'.jpg'
            if name in self.train_original_mismatched:
                # Skip images marked as mismatched
                continue
            assert name[:3] in ['pos','neg']
            if name[:3] == 'pos':
                y = 1
            else:
                y = 0
            meta = {
                'full_name': name,
                'filename': image_name,
                'coordinates': name.split('_')[4],
                'counts':  int(name.split('_')[2].split('c')[0]),
            }
            images.append({'x': (lambda filename: lambda: self.load(filename))(filename),
                           'm': meta,
                           'y': y})
        return images
        
    def load_original_images(self, dataset = "train"):
        """
        Load the data
        """
        
        images = []

        if dataset == "train":
            logger.debug('Loading train set original images')

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
            logger.debug('Loading stage 1 test set original images')

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
        #logger.debug('Loading image from disk: %s' % filename)
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

    def reset(self):
        self.batch_index = 0

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
    def __init__(self, data, data_transformation, batch_size, shuffle, seed):
        self.data = data
        self.data_transformation = data_transformation
        super(DataIterator, self).__init__(len(data), batch_size, shuffle, seed)

    def next(self):
        # Only keep index advancing under lock
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_x = None
        batch_y = None

        for i, j in enumerate(index_array):
            # i is the batch index (0 to batch_size)
            # j is the index of the data to put at this batch index
            d = dict(self.data[j]) # Get the dictionary of this individual sample from the batch (and make a shallow copy of it)
            if self.data_transformation is not None:
                d = self.data_transformation.apply(d)
            if batch_x is None:
                batch_x = np.zeros(tuple([current_batch_size] + list(d['x'].shape)), dtype=K.floatx())
                    
            batch_x[i] = d['x']

            if 'y' in d:
                if batch_y is None:
                    batch_y = np.zeros(current_batch_size)

                batch_y[i] = d['y']
        batch_x = preprocess_input(batch_x)
        if batch_y is not None:
            return batch_x, batch_y
        else:
            return batch_x

class Transformer(object):
    """
    Abstract class for chainable data transformations.
    """

    def __init__(self, next = None):
        self.next = next

    def chain(self, dataTransformation):
        if self.next is None:
            self.next = dataTransformation
        else:
            self.next.chain(dataTransformation)
        return self

    def apply(self, data):
        d = self._transform(data)
        if self.next is not None:
            d = self.next.apply(d)
        return d

    @abc.abstractmethod
    def _transform(self, data):
        pass

class IdentityTransformer(Transformer):
    """
    The identity transformation.
    """

    def _transform(self, data):
        return data

class ResizeTransformer(Transformer):

    def __init__(self, shape, *args, **kwargs):
        super(ResizeTransformer, self).__init__(*args, **kwargs)

        self.shape = shape

    def _transform(self, data):
        data['x'] = scipy.misc.imresize(data['x'], self.shape)
        return data

class LoadTransformer(Transformer):
    """
    Transformation for the initial loading of the data
    """
    def _transform(self, data):
        data['x'] = data['x']()
        return data

class AugmentationTransformer(Transformer):
    """
    Data augmentor augmentation.
    """
    def __init__(self, store_original = False, *args, **kwargs):
        super(AugmentationTransformer, self).__init__(*args, **kwargs)

        import augmentor
        from keras.preprocessing.image import ImageDataGenerator

        self.store_original = store_original

        imagegen = ImageDataGenerator(
                rescale = None,
                #rotation_range = settings.AUGMENTATION_ROTATION_RANGE,
                #shear_range = settings.AUGMENTATION_SHEAR_RANGE,
                #zoom_range = settings.AUGMENTATION_ZOOM_RANGE,
                #width_shift_range = settings.AUGMENTATION_WIDTH_SHIFT_RANGE,
                #height_shift_range = settings.AUGMENTATION_HEIGHT_SHIFT_RANGE,
                horizontal_flip = settings.AUGMENTATION_HORIZONTAL_FLIP,
                vertical_flip = settings.AUGMENTATION_VERTICAL_FLIP
                #channel_shift_range = settings.AUGMENTATION_CHANNEL_SHIFT_RANGE
                )  
        
        self.augm = augmentor.Augmentor(imagegen) 

    def _transform(self, data):
        if self.store_original: 
            data['m']['orig_x'] = data['x']

        data['x'] = self.augm.augment(data['x'])
        return data

