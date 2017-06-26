"""
Module containing the data loading functionality.
"""

import abc
import collections
import csv
import glob
import os
import threading
import itertools
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

    def load_test_coordinates(self):
        """
        Load the coordinates CSV for the test dataset.
        Turn it into a dictionary of lists of coordinates.
        
        {image_id: [
                {
                    x_coord: n
                    y_coord: n
                }
            ]
        }

        :return: A dictionary of lists of coordinates.
        """
        logger.debug('Loading test image coordinates')
        d = collections.defaultdict(list)
        with open(settings.TEST_COORDINATES_CSV, 'r') as file:
            [d[utils.get_file_name_part(row['filename'])].append(utils.remove_key_from_dict(row, '', 'filename')) for row in csv.DictReader(file)]
        return dict(d)
        
    def _load_heatmap_crop_images(self):
        import cropping
        """
        Loads the heatmap crops and generates the object densitiy maps (odm)
        """
        odm_original_size = 400
        odm_target_size = 80
        skip_pups = True
        i = 0
        #Build the type of marks for each type of sealion
        marks = {
            'adult_males':    utils.get_gaussian_mark(3.),
            'subadult_males': utils.get_gaussian_mark(3.),
            'juveniles':      utils.get_gaussian_mark(2.5),
            'pups':           utils.get_gaussian_mark(0.7),
            'adult_females':  utils.get_gaussian_mark(2.5)
        }
        images = []
        filepaths = glob.glob(os.path.join(settings.TRAIN_HEATMAP_DIR,'*.jpg'))
        if 0:
            #for debug
            filepaths = filepaths[:100000]
            settings.logger.warning("Not using all the crops")
        total = len(filepaths)
        logger.info("Generating object density maps of size "+str(odm_target_size)+" for "+str(total)+" crops...")
        logger.warning("Skip_pups set to "+str(skip_pups))
        #Iterate over all the crops
        for filepath in filepaths:
            meta = {}
            meta['filepath'] = filepath
            meta['filename'] = utils.get_file_name_part(meta['filepath'])
            meta['count'] = int(meta['filename'].split('clions')[0])
            meta['coordinates'] = meta['filename'].split('_')[1][2:]
            meta['id'] = meta['filename'].split('in')[1].split('_')[0]
            if meta['count'] == 0 and random.choice([0,0,1,1,1]):
                #We skip 60% of the negatives
                total -= 1
                continue
            #Initialize the object density map matrix
            odm = np.zeros((odm_original_size,odm_original_size))
            #Fill the odm with marks where the sealions are
            for sealion in self.train_original_coordinates[meta['id']]:
                if sealion['category'] == 'pups' and skip_pups:
                    continue
                sealion['row'] = float(sealion['y_coord'])
                sealion['column'] = float(sealion['x_coord'])
                crop_ix = {
                    'row': float(meta['coordinates'].split('-')[1]),
                    'column': float(meta['coordinates'].split('-')[0])
                }
                if cropping.RegionCropper.is_inside(None, sealion, crop_ix, odm_original_size):
                    sealion['column'] = sealion['column'] - crop_ix['column']
                    sealion['row'] = sealion['row'] - crop_ix['row']

                    row = int(sealion['row'])
                    column = int(sealion['column'])
                    mark = marks[sealion['category']]
                    radius = round(mark.shape[0]/2.)
                    effective_mark = mark[max(0,radius-row):radius + odm_original_size - row, max(0,radius-column):radius + odm_original_size - column]
                    odm[max(0,row-radius):row+radius,max(0,column-radius):column+radius] += effective_mark
            #Resize to match the desired input shape of the network
            odm = scipy.misc.imresize(odm,(odm_target_size,odm_target_size))
            if odm.max() > 0:
                odm = odm/odm.max()     
            #Add one dimension for the single channel
            odm = np.expand_dims(odm, axis = 2)
            #print(odm.max(), odm.mean(),9999)
            images.append({'x': (lambda filepath: lambda: self.load(filepath))(meta['filepath']),
               'm': meta,
               'y': odm})
            if i%1000==0:
                logger.info(str(100*i/total)[:5]+str("% completed"))
            i += 1
        return images

    def _load_region_crop_images(self):
        images = []
        crops_dir = settings.REGION_CROPS_DIR

        # Get all images
        filenames_pos = sorted(glob.glob(os.path.join(crops_dir,'pos',"*.jpg")))
        filenames_neg = sorted(glob.glob(os.path.join(crops_dir,'neg',"*.jpg")))
        filenames = filenames_pos + filenames_neg
        for filename in filenames:
            #10clions_at40-1680_in66_400px
            #0clions_at0-2471_in579_400px
            name = utils.get_file_name_part(filename)
            #image_name = name.split('_')[1].split('d')[1]
            image_name = name.split('in')[1].split('_')[0]
            if image_name in self.train_original_mismatched:
                # Skip images marked as mismatched
                continue
            
            y = filename.split(os.path.sep)[-2]
            assert y in ['pos','neg']
            if y == 'pos':
                y = 'positive'
            else:
                y = 'negative'
            meta = {
                'full_name': name,
                'filename': image_name,
                'coordinates': name.split('_')[1][2:],
                'counts':  int(name.split('clions')[0]),
            }
            
            images.append({'x': (lambda filename: lambda: self.load(filename))(filename),
                           'm': meta,
                           'y': y})
        return images

    def _load_sea_lion_crop_images(self):
        images = []
        crops_dir = settings.SEA_LION_CROPS_DIR

        # Get all images
        filenames = sorted(glob.glob(os.path.join(crops_dir,"*.jpg")))
        for filename in filenames:
            #adult_males_id923_1clions_at_1944-1425_197px
            #negative_id0_0clions_at_13-913_197px
            name = utils.get_file_name_part(filename)
            
            name_parts = name.split('_id')
            clss = name_parts[0]
            image_name = name_parts[1].split('_')[0]

            if image_name in self.train_original_mismatched:
                # Skip images marked as mismatched
                continue
            
            meta = {
                'full_name': name,
                'filename': image_name
            }
            
            images.append({'x': (lambda filename: lambda: self.load(filename))(filename),
                           'm': meta,
                           'y': clss})
        return images

    def load_crop_images(self, data_type):
        """
        Load precropped data
        """
        
        assert data_type in ['sea_lion_crops', 'region_crops','heatmap_crops']
        if data_type == 'region_crops':
            logger.debug('Loading region crop images')
            images = self._load_region_crop_images()
        elif data_type == 'sea_lion_crops':
            logger.debug('Loading sea lion crop images')
            images = self._load_sea_lion_crop_images()
        else:
            logger.debug('Loading heatmap crop images')
            images = self._load_heatmap_crop_images()
            
        logger.debug('Loaded %s images' % len(images))
        return images

    def load_density_map_feature_crops(self):
        """
        Load density map feature patches.

        The output is:

        [ # A list of dicts for each unique image patch, containing all features corresponding to that patch
            {
                'features': { # Feature bank; a dictionary that groups feature types together (e.g., all LOGs are grouped)
                    <feature name>: { # A dictionary mapping from specific feature type settings to feature images
                        <feature setting>: <function to load feature image>
                    }
                },
                'meta': {
                    'image_name': <image id>,
                    'patch': { # Patch coordinates
                        'x': <left x coordinate>,
                        'y': <top y coordinate>,
                        'width': <width>,
                        'height': <height>
                    },
                    'coordinates': [ # A list of sea lion coordinates within the patch, with coordinates relative to the patch
                        {
                            'x': <x coordinate>,
                            'y': <y coordinate>,
                            'category': <sea lion type>
                        }
                    ]
                }
            }
        ]
        """

        logger.debug('Loading density map features')

        images = {}
        crops_dir = settings.DENSITY_MAP_FEATURE_CROPS_DIR

        # Get all images
        #filenames = sorted(glob.glob(os.path.join(crops_dir,"*.png")))
        filenames = sorted([f for f in os.listdir(crops_dir) if f[-4:] == ".png"])
        n = 0
        for filename in filenames:
            # <image id>_<crop x coordinate>-<crop y coordinate>-<crop width>-<crop height>_<feature name>-<feature setting>.jpg
            name = utils.get_file_name_part(filename)
            name_parts = name.split('_')

            image_id = name_parts[0]
            coordinate_parts = name_parts[1].split('-')
            feature_parts = name_parts[2].split('-')

            bounding_box = {
                'x': 2 * int(coordinate_parts[0]), 
                'y': 2 * int(coordinate_parts[1]), 
                'width': 2 * int(coordinate_parts[2]), 
                'height': 2 * int(coordinate_parts[3])}
            feature_name = feature_parts[0]
            feature_setting = feature_parts[1]

            key = str((image_id, bounding_box.values()))

            if image_id in self.train_original_mismatched:
                # Skip images marked as mismatched
                continue

            n += 1

            # Add image patch if it does not exist yet
            if key not in images:
                # Get coordinates of sea lions in the original full-size image
                orig_coordinates = self.train_original_coordinates[image_id] if image_id in self.train_original_coordinates else []

                # Get all sea lion coordinates that are within (or very close to) this patch and transform coordinates to the patch coordinate base
                coordinates = []
                for coord in orig_coordinates:
                    x = int(float(coord['x_coord']))
                    y = int(float(coord['y_coord']))
                    if (
                            bounding_box['x'] - 150 <= x < bounding_box['x'] + bounding_box['width'] + 150
                        and bounding_box['y'] - 150 <= y < bounding_box['y'] + bounding_box['height'] + 150):
                        coordinates.append({
                            'x_coord': x - bounding_box['x'],
                            'y_coord': y - bounding_box['y'],
                            'category': coord['category']})

                images[key] = {
                    'features': {}, 
                    'meta': {
                        'image_name': image_id,
                        'patch': bounding_box,
                        'coordinates': coordinates
                    }
                }

            # Add feature group if it does not exist yet
            if feature_name not in images[key]['features']:
                images[key]['features'][feature_name] = {}
            
            # Add feature
            images[key]['features'][feature_name][feature_setting] = (lambda filename: lambda: self.load(filename))(os.path.join(crops_dir, filename))
            
        # Turn into list
        images = [img for img in images.values()]

        logger.debug('Loaded %s features for %s images' % (n, len(images)))

        return images

    def load_full_size_feature_images(self, dataset = "train"):
        """
        Load full size density map features.

        The output is:

        [ # A list of dicts for each unique original image, containing all features corresponding to that image
            {
                'features': { # Feature bank; a dictionary that groups feature types together (e.g., all LOGs are grouped)
                    <feature name>: { # A dictionary mapping from specific feature type settings to feature images
                        <feature setting>: <function to load feature image>
                    }
                },
                'meta': {
                    'image_name': <image id>,

                    # train only:
                    'coordinates': [ # A list of sea lion coordinates within the original image
                        {
                            'x': <x coordinate>,
                            'y': <y coordinate>,
                            'category': <sea lion type>
                        }
                    ],
                    'counts': <total categorized count of sea lions in the image>
                }
            }
        ]
        """

        images = {}

        if dataset == "train":
            logger.debug("Loading train set full-size feature images")
            features_dir = settings.TRAIN_FEATURES_DIR
            train = True

        elif dataset == "test_st1":
            logger.debug("Loading test set full-size feature images")
            features_dir = settings.TEST_FEATURES_DIR
            train = False

        # Get all images
        filenames = glob.glob(os.path.join(features_dir,"*.png"))

        for filename in filenames:
            # <image id>_<feature name>-<feature setting>.jpg
            name = utils.get_file_name_part(filename)
            name_parts = name.split('_')

            image_id = name_parts[0]
            feature_parts = name_parts[1].split('-')

            feature_name = feature_parts[0]
            feature_setting = feature_parts[1]

            if image_id in self.train_original_mismatched:
                # Skip images marked as mismatched
                continue

            # Add base image if it does not exist yet
            if image_id not in images:
                if train:
                    meta = {
                        'image_name': image_id,
                        'coordinates': self.train_original_coordinates[name] if name in self.train_original_coordinates else [],
                        'counts': self.train_original_counts[name]
                    }
                else:
                    meta = {
                        'image_name': image_id
                    }

                images[image_id] = {
                    'features': {}, 
                    'meta': meta
                }

            # Add feature group if it does not exist yet
            if feature_name not in images[image_id]['features']:
                images[image_id]['features'][feature_name] = {}
            
            # Add feature
            images[image_id]['features'][feature_name][feature_setting] = (lambda filename: lambda: self.load(filename))(filename)

        # Turn into list
        images = [img for img in images.values()]
        images = sorted(images, key = lambda img: img['meta']['image_name'])

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
                    'counts': self.train_original_counts[name]
                }

                images.append({'x': (lambda filename: lambda: self.load(filename))(filename),
                               'm': meta})

        elif dataset == "test_st1":
            logger.debug('Loading stage 1 test set original images')

            # Get all test original images
            filenames = sorted(glob.glob(os.path.join(settings.TEST_ORIGINAL_IMAGES_DIR, "*.jpg")))
            for filename in filenames:
                name = utils.get_file_name_part(filename)

                #if name in self.train_original_mismatched:
                #    # Skip images marked as mismatched
                #    continue

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
    def __init__(self, n, batch_size, shuffle, seed, class_balancing = False, classes = None):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        if class_balancing:
            self.index_generator = self._balanced_flow_index(classes, batch_size, shuffle, seed)
        else:
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

    def __index_generator(self, indices, shuffle=False, seed=None):
        i = 0
        while 1:
            i = i + 1
            if seed is not None:
                np.random.seed(seed + 1)

            if shuffle:
                indices = np.random.permutation(indices)

            for idx in indices:
                yield idx

    def _balanced_flow_index(self, classes, batch_size=32, shuffle=False, seed=None):
        n = len(classes)
        unique_classes = set(classes)
        generators = []
        class_to_indices = {}

        # Create a map from classes to lists of data indices belonging to those classes
        for clss, idx in zip(classes, itertools.count()):
            if clss not in class_to_indices:
                class_to_indices[clss] = []
            class_to_indices[clss].append(idx)

        # Create infinite generators to generate indices of classes
        for clss in unique_classes:
            generators.append(self.__index_generator(class_to_indices[clss], shuffle=shuffle, seed=seed))

        # Create a generator to sequentially get an index from each of the generators
        def gen_from_generators():
            while 1:
                for generator in generators:
                    yield next(generator)

        # Ensure self.batch_index is 0.
        self.reset()

        current_idx = 0
        current_batch = []
        for idx in gen_from_generators():
            current_idx += 1

            current_batch.append(idx)
            current_batch_size = len(current_batch)

            if current_batch_size == batch_size:
                yield current_batch, current_idx % n, current_batch_size
                current_batch = []

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

class DataIterator(Iterator):
    def __init__(self, data, data_transformation, batch_size, shuffle, seed, class_balancing = False, class_transformation = lambda x: x):
        self.data = data
        self.data_transformation = data_transformation
        self.class_transformation = class_transformation
        if class_balancing:
            y = [self.class_transformation(d['y']) for d in data]
        else:
            y = None
        super(DataIterator, self).__init__(len(data), batch_size, shuffle, seed, class_balancing = class_balancing, classes = y)

    def next(self):
        # Only keep index advancing under lock
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_x = None
        batch_y = None
        batch_m = None
        
        for i, j in enumerate(index_array):
            # i is the batch index (0 to batch_size)
            # j is the index of the data to put at this batch index
            d = dict(self.data[j]) # Get the dictionary of this individual sample from the batch (and make a shallow copy of it)
            if self.data_transformation is not None:
                d = self.data_transformation.apply(d)
            if batch_x is None:
                batch_x = np.zeros(tuple([current_batch_size] + list(d['x'].shape)), dtype=K.floatx())
            if batch_m is None:
                batch_m = list()
                
            batch_x[i] = d['x']

            if 'm' in d:
                batch_m.append(d['m'])
            if 'y' in d:
                if batch_y is None:
                    #Check what kind of Y are we generating
                    if type(d['y']) == np.ndarray and len(d['y'].shape) > 1:
                        #Generating heatmaps
                        batch_y = np.zeros((current_batch_size, d['y'].shape[0], d['y'].shape[1], 1))
                    else:
                        #Not generating heatmaps
                        batch_y = np.zeros(current_batch_size)
                batch_y[i] = self.class_transformation(d['y'])
        #batch_x = preprocess_input(batch_x)
        if batch_y is not None:
            #For now don't return batch_m in this case
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

class RescaleTransformer(Transformer):

    def __init__(self,  *args, **kwargs):
        super(RescaleTransformer, self).__init__(*args, **kwargs)

    def _transform(self, data):
        if data['x'].max() > 0:
            data['x'] = data['x'] / data['x'].max()
        return data

class LoadTransformer(Transformer):
    """
    Transformation for the initial loading of the data
    """
    def _transform(self, data):
        data['x'] = data['x']()
        return data

class LoadDensityFeatureTransformer(Transformer):
    """
    Transformation for the initial loading of the density map feature data
    """
    def _transform(self, data):
        features = []
        features.append(data['features']['dog']['0.7']())
        features.append(data['features']['dog']['3.5']())
        features.append(data['features']['dog']['7.5']())
        features.append(data['features']['dog']['15']())

        features.append(data['features']['gs']['0.7']())
        features.append(data['features']['gs']['3.5']())
        features.append(data['features']['gs']['7.5']())
        features.append(data['features']['gs']['15']())

        features.append(data['features']['hoge1']['0.7']())
        features.append(data['features']['hoge1']['3.5']())
        features.append(data['features']['hoge1']['7.5']())
        features.append(data['features']['hoge1']['15']())

        features.append(data['features']['hoge2']['0.7']())
        features.append(data['features']['hoge2']['3.5']())
        features.append(data['features']['hoge2']['7.5']())
        features.append(data['features']['hoge2']['15']())

        features.append(data['features']['ste1']['0.7']())
        features.append(data['features']['ste1']['3.5']())
        features.append(data['features']['ste1']['7.5']())
        features.append(data['features']['ste1']['15']())


        shapes = list(map((lambda f: f.shape if len(f.shape) == 3 else (f.shape[0], f.shape[1], 1)), features))
        numChannels = sum(map((lambda shape: shape[2]), shapes))

        concat = np.zeros((shapes[0][0], shapes[0][1], numChannels))

        channelsSeen = 0
        for feature, shape in zip(features, shapes):
            if len(feature.shape) < 3:
                feature = np.expand_dims(feature, axis=2)
            concat[..., channelsSeen:(channelsSeen + shape[2])] = feature
            channelsSeen += shape[2]

        data['x'] = concat
        return data

class CreateDensityMapTransformer(Transformer):
    """
    Create the actual density map for the density map data
    """
    def __init__(self, sigma_per_class = None, scale = 1.0, *args, **kwargs):
        super(CreateDensityMapTransformer, self).__init__(*args, **kwargs)

        self.sigma_per_class = sigma_per_class
        self.scale = scale

    def _transform(self, data):
        import math
        
        coords = data['meta']['coordinates']
        m = utils.sea_lion_density_map(
            math.ceil(data['meta']['patch']['width'] * self.scale),
            math.ceil(data['meta']['patch']['height'] * self.scale),
            data['meta']['coordinates'],
            sigma = 35,
            sigma_per_class = self.sigma_per_class,
            scale = self.scale)

        # Add dimension such that m is of shape (width, height, 1)
        data['y'] = np.expand_dims(m, axis=2)
        return data

class SyncedAugmentationTransformer(Transformer):
    """
    Data augmentor augmentation used for training on heatmaps.
    This is class is needed because in heatmap training the augmentations
    on X needs to be done on Y as well, in a synchronous manner
    """
    def __init__(self, store_original = False, *args, **kwargs):
        super(SyncedAugmentationTransformer, self).__init__(*args, **kwargs)

        self.store_original = store_original
        
    def random_synced_flip(self, data):  
        flipud = random.choice([True, False])
        fliplr = random.choice([True, False])
        if flipud:
            data['x'] = np.flipud(data['x'])
            data['y'] = np.flipud(data['y'])
        if fliplr:
            data['x'] = np.fliplr(data['x'])
            data['y'] = np.fliplr(data['y'])
        return data
    
    def _transform(self, data):
        if self.store_original: 
            data['m']['orig_x'] = data['x']
            data['m']['orig_y'] = data['y']
        
        data = self.random_synced_flip(data)
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

def sea_lion_type_to_sea_lion_or_not(sea_lion_label):
    if sea_lion_label == 'negative':
        return 0
    else:
        return 1
