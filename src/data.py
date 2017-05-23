"""
Module containing the data loading functionality.
"""

import abc
import collections
import csv
import glob
import os
import threading

import scipy
import scipy.misc
import numpy as np
import sklearn.model_selection
import pandas as pd

from keras import backend as K

import settings
import utils
import random
from ast import literal_eval


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

class Cropper:
    """
    Class for cropping images
    """
    def __init__(self, crop_size, total_crops, pos_perc, min_sealions_herd):
        """
        :param crop_size: The size of the desired crop        
        """
        self.crop_size = crop_size
        self.total_crops = total_crops
        self.loader = Loader()
        self.pos_perc = pos_perc
        self.min_sealions_herd = min_sealions_herd
        self.positive_crops = []
        self.negative_crops = []
        print("Cropper intialisated with:\n\tcrop_size",self.crop_size,"\n\ttotal_crops",total_crops,"\n\tpos_perc",pos_perc)
        return
        
    def crop_image(self, image, coordinates):    
        """
        Returns a square shaped crop from an image.
        
        :param image: The image from which to take the crop
        :param coordinates: Tuple (x, y) that contains the left upper corner coordinates of the crop
        :param crop_size: The size of the desired crop
        """
        x_coordinate, y_coordinate = coordinates[0], coordinates[1]
        return image[y_coordinate : y_coordinate + self.crop_size, x_coordinate : x_coordinate + self.crop_size, :]
    
    def is_inside(self, xy_sealion, xy_crop):
        """
        Returns true if the sea lion coordinate xy_sealion is inside the crop defined by
        xy_crop and self.crop_size
        """
        is_in_x_axis = xy_crop[0] < xy_sealion[0] < xy_crop[0] + self.crop_size
        is_in_y_axis = xy_crop[1] < xy_sealion[1] < xy_crop[1] + self.crop_size

        return is_in_x_axis and is_in_y_axis
    
    def random_crop(self, train_id):
        """
        Randomly select and returns a crop point on image train_id
        """
        size = literal_eval(self.loader.train_original_counts[train_id]['size'])
        x_coordinate = random.randint(0,size[1] - self.crop_size)
        y_coordinate = random.randint(0,size[0] - self.crop_size)
        xy_crop = (x_coordinate, y_coordinate)
        return xy_crop
    
    def remove_duplicate_crops(self):
        self.positive_crops = list(set(self.positive_crops))
        self.negative_crops = list(set(self.negative_crops))
        
    def enough_positives(self):
        return len(self.positive_crops)>=self.pos_perc*self.total_crops
    
    def enough_negatives(self):
        return len(self.negative_crops)>=(1-self.pos_perc)*self.total_crops
    
    def enough_crops(self):
        return self.enough_positives() and self.enough_negatives()
    
    def count_sealions_in_crop(self, xy_crop, train_id, skip_pups):
        """
        Counts how many sea lions are inside the crop. Pups can be skipped
        since the look like rocks and usually are not alone
        """
        count = 0
        if train_id not in self.loader.train_original_coordinates.keys():
            return count
        for coordinate in self.loader.train_original_coordinates[train_id]:
            xy_sealion = (round(float(coordinate['x_coord'])), round(float(coordinate['y_coord'])))
            if skip_pups and coordinate['category'] == 'pups':
                continue
            if self.is_inside(xy_sealion, xy_crop):
                count += 1
        return count

    def find_crops(self):
        """
        Find negative and positive crops in the whole train set
        """
        train_ids = list(self.loader.train_original_counts.keys())

        self.trials = 0
        while not self.enough_crops():
            train_id = random.choice(train_ids)
            if train_id in self.loader.train_original_mismatched:
                # Skip images marked as mismatched
                continue
            xy_crop = self.random_crop(train_id)
            #print(xy_crop)
            n_sealions = self.count_sealions_in_crop(xy_crop, train_id, skip_pups = True)
            if n_sealions > self.min_sealions_herd:
                if not self.enough_positives():
                    self.positive_crops.append((xy_crop,train_id,n_sealions,'pos'))
            else:
                if not self.enough_negatives():
                    self.negative_crops.append((xy_crop,train_id,n_sealions,'neg'))
            self.trials += 1
            if self.trials % 5000 == 0:
                self.remove_duplicate_crops()
                print(self.trials," trials so far, ",len(self.positive_crops),"positives and",len(self.negative_crops),"negatives")
        print("Finished after ",self.trials,"trials")
        
    def show_some_crops(self,n = 10):
        """
        Visualize some of the positive and negative crops that the class is taking
        """
        for i in range(n):
            pc = self.positive_crops[i]
            img = self.loader.load(os.path.join(settings.TRAIN_ORIGINAL_IMAGES_DIR, pc[1]+'.jpg'))
            c = self.crop_image(img,pc[0])
            plt.subplot(1,2,1)
            plt.title(pc[2])
            plt.imshow(c)
            nc = self.negative_crops[i]
            img = self.loader.load(os.path.join(settings.TRAIN_ORIGINAL_IMAGES_DIR, nc[1]+'.jpg'))
            c = self.crop_image(img,nc[0])
            plt.subplot(1,2,2)
            plt.title(nc[2])
            plt.imshow(c)
            plt.show()
            
    def save_crops(self):
        """
        Save crops to disk
        """
        # Create weight output dir if it does not exist
        if not os.path.exists(settings.CROPS_OUTPUT_DIR):
            os.makedirs(settings.CROPS_OUTPUT_DIR)       
        crops = self.positive_crops + self.negative_crops
        crops = sorted(crops, key = lambda x: x[1])
        train_id = ''
        print("Attempting to write ",len(crops)," to ",settings.CROPS_OUTPUT_DIR)
        count = 0
        for crop_meta in crops:
            if crop_meta[1] != train_id:
                train_id = crop_meta[1]
                image = self.loader.load(os.path.join(settings.TRAIN_ORIGINAL_IMAGES_DIR,train_id+'.jpg'))
            crop_image = self.crop_image(image, crop_meta[0])
            self.save_crop(crop_image,crop_meta)
            count += 1
            if count % 5 == 0:
                print(len(crops)-count,"crops left")
        print(count,"crops were saved in",settings.CROPS_OUTPUT_DIR)
            
    def save_crop(self, crop_image, crop_meta):
        """
        Save one crop to disk, the filename has some metadata information
        and looks like this 'pos_id402_14clions_at_1018-1378_224px.jpg'
        """
        filename = crop_meta[3]+'_id'+crop_meta[1]+'_'+str(crop_meta[2])+'clions_at_'+str(crop_meta[0][0])+'-'+str(crop_meta[0][1])+'_'+str(self.crop_size)+'px.jpg'
        scipy.misc.imsave(os.path.join(settings.CROPS_OUTPUT_DIR,filename), crop_image)

 

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
            d = self.data[j]
            if self.data_transformation is not None:
                d = self.data_transformation.apply(d)
            if batch_x is None:
                batch_x = np.zeros(tuple([current_batch_size] + list(d['x'].shape)), dtype=K.floatx())
                    
            batch_x[i] = d['x']

            if 'y' in d:
                if batch_y is None:
                    batch_y = np.zeros(current_batch_size)

                batch_y[i] = d['y']

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
                rotation_range = settings.AUGMENTATION_ROTATION_RANGE,
                shear_range = settings.AUGMENTATION_SHEAR_RANGE,
                zoom_range = settings.AUGMENTATION_ZOOM_RANGE,
                width_shift_range = settings.AUGMENTATION_WIDTH_SHIFT_RANGE,
                height_shift_range = settings.AUGMENTATION_HEIGHT_SHIFT_RANGE,
                horizontal_flip = settings.AUGMENTATION_HORIZONTAL_FLIP,
                vertical_flip = settings.AUGMENTATION_VERTICAL_FLIP,
                channel_shift_range = settings.AUGMENTATION_CHANNEL_SHIFT_RANGE
                )  
        
        self.augm = augmentor.Augmentor(imagegen) 

    def _transform(self, data):
        if self.store_original: 
            data['m']['orig_x'] = data['x']

        data['x'] = self.augm.augment(data['x'])
        return data

