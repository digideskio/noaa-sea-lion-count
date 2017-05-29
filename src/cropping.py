"""
Module for data cropping functionality.
"""

import os

import scipy
import numpy as np

import ast
import random

import math
import time
import settings
import data

from matplotlib import pyplot as plt
#%matplotlib inline

logger = settings.logger.getChild('cropping')

class RegionCropper:
    """
    Class for cropping images
    """
    def __init__(self, crop_size, attention = False):
        """        
        :param crop_size: Int, the size of the desired crops
        :param attention: Boolean, whether to apply blackout attention to the crops
            or not
        """
        self.crop_size = crop_size
        self.attention = attention
        
        
        self.loader = data.Loader()
               
        self.window_coords = {'row':0, 'column':0}
        self.current_image_id = None
        self.current_image_size = {'x': None, 'y': None}
        self.current_image = None
        
        self.candidate_matrix = None
        self.candidate_current_ix = {'row': 0, 'column': 0}
        
    def init_candidate_matrix(self):
        """
        Restarts the candidate matrix taking into account the size of the
        current original image
        """
        columns = int((self.current_image_size['x'] - self.crop_size) / self.sliding_px) + 1
        rows = int((self.current_image_size['y'] - self.crop_size) / self.sliding_px) + 1
        self.candidate_matrix = np.zeros((rows,columns)).astype('int32')
        self.candidate_current_ix = {'row': 0, 'column': 0}
        
        
    def count_sealions_in_crop(self, crop_ix, skip_pups = True):
        """
        Counts how many sea lions are inside the crop. Pups can be skipped
        since the look like rocks and usually are not alone
        """
        count = 0
        if self.current_image_id not in self.loader.train_original_coordinates.keys():
            return count
        for coordinate in self.loader.train_original_coordinates[self.current_image_id]:
            sealion = {'column': round(float(coordinate['x_coord'])), 'row': round(float(coordinate['y_coord']))}
            if skip_pups and coordinate['category'] == 'pups':
                continue
            if self.is_inside(sealion, crop_ix):
                count += 1
        return count 
    
    def is_inside(self, sealion, crop_ix):
        """
        Returns true if the sea lion coordinate xy_sealion is inside the crop defined by
        xy_crop and self.crop_size
        """
        is_in_x_axis = crop_ix['column'] < sealion['column'] < crop_ix['column'] + self.crop_size
        is_in_y_axis = crop_ix['row'] < sealion['row'] < crop_ix['row'] + self.crop_size
        return is_in_x_axis and is_in_y_axis
    
    def is_positive(self, crop_ix):
        """
        Returns True if the current crop (indicated by self.window_coords and self.crop_size)
        is positive aka contains more than self.min_sealions_pos sealions
        """
        sealions_count = self.count_sealions_in_crop(crop_ix, skip_pups = True)
        return sealions_count >= self.min_sealions_pos, sealions_count
    

    def eval_current_image_size(self):
        size_eval = ast.literal_eval(self.loader.train_original_counts[self.current_image_id]['size'])
        self.current_image_size = {'x': size_eval[1], 'y': size_eval[0]}
        
    def print_candidate_matrix(self):
        """
        Plots the current state of the candidate matrix. It is useful
        to monitor the sampling process
        """
        plt.figure()
        plt.imshow(self.candidate_matrix, cmap = 'gray')
        plt.title(str(self.candidate_matrix.sum())+' candidates left')
        plt.show()
        
    def crop_from_current_image(self, crop_ix):    
        """
        Returns a square shaped crop from an image.
        
        :param image: The image from which to take the crop
        :param coordinates: Tuple (x, y) that contains the left upper corner coordinates of the crop
        :param crop_size: The size of the desired crop
        """
        image_crop = self.current_image[crop_ix['row'] : crop_ix['row'] + self.crop_size, crop_ix['column'] : crop_ix['column'] + self.crop_size, :]
        return image_crop
    
    def plot_crop_ix(self, crop_ix):
        """
        Receives coordinates of a crop that belongs to the current original image
        and plots it
        """
        image_crop = self.crop_from_current_image(crop_ix)
        plt.figure()
        plt.imshow(image_crop)
        plt.title(str(image_crop.shape))
        plt.show()
        if image_crop.shape[0]<self.crop_size:
            logger.info("Something weird happenned")
            plt.figure()
            plt.imshow(self.current_image)
            plt.title(str(crop_ix)+' '+self.current_image_id)
            plt.show()
        
    def candidate_ix_to_image_ix(self, candidate_ix):
        """
        Takes the coordinates of en element of the candidate matrix and gives
        its respective coordinates in the orignal image
        """
        crop_ix = {}
        crop_ix['column'] = candidate_ix['column'] * self.sliding_px
        crop_ix['row'] = candidate_ix['row'] * self.sliding_px
        return crop_ix
            
    def overlap(self, crop_ix_a, crop_ix_b):
        """
        UNUSED
        Computes the overlap between two crops of an image. Used for debbuging
        but not for production
        """
        x = max(crop_ix_a['x'], crop_ix_b['x'])
        y = max(crop_ix_a['y'], crop_ix_b['y'])
        w = max(0, min(crop_ix_a['x'] + self.crop_size, crop_ix_b['x'] + self.crop_size) - x)
        h = max(0, min(crop_ix_a['y'] + self.crop_size, crop_ix_b['y'] + self.crop_size) - y)
        intersection = w*h
        overlap = intersection / (crop_size**2)
        return onverlap
    
    def slide_window(self):
        """
        Slides the window i.e. self.window_coords gets changed accordingly. The sliding process starts in the left upper corner
        of the image and moves all the way to the right, then slides down and starts from left to right again.
        Needs to take into account what is the current image (self.current_image_id) and more specifically the
        current image size.
        """

        if self.window_coords['column'] + self.crop_size + self.sliding_px > self.current_image_size['x']:
            self.window_coords['column'] = 0
            self.candidate_current_ix['column'] = 0
            if self.window_coords['row'] + self.crop_size + self.sliding_px > self.current_image_size['y']:
                self.window_coords['row'] = 0
                self.candidate_current_ix['row'] = 0
            else:
                self.window_coords['row'] += self.sliding_px
                self.candidate_current_ix['row'] += 1
        else:
            self.window_coords['column'] += self.sliding_px
            self.candidate_current_ix['column'] += 1
        
        sliding_finished = self.window_coords['column'] == 0 and self.window_coords['row'] == 0
        
        return sliding_finished
        

    def find_pos_crops_current_image(self):
        """
        Finds positive crops in the current image using a sliding window approach and
        the overlapping criteria
        """
        sliding_finished = False
        try:
            coordinates = self.loader.train_original_coordinates[self.current_image_id]
        except KeyError:
            logger.info(self.current_image_id+" has not sealions")
            sliding_finished = True
        while not sliding_finished:
            is_positive, sealions_count = self.is_positive(self.window_coords)
            if is_positive:
                row_ix = self.candidate_current_ix['row']
                column_ix = self.candidate_current_ix['column']
                self.candidate_matrix[row_ix][column_ix] = 1              
            sliding_finished = self.slide_window()            
        return
    
    def select_and_save_candidates(self):
        """
        Randomly samples positive elements from the candidate matrix (that must have
        already built)
        """
        if not os.path.exists(os.path.join(settings.CROPS_OUTPUT_DIR,'pos')):
            os.makedirs(os.path.join(settings.CROPS_OUTPUT_DIR,'pos'))  
        radius = math.ceil(self.crop_size * self.max_overlap_perc / self.sliding_px)
        crops_count = 0
        while self.candidate_matrix.sum() != 0:
            if self.plot:
                self.print_candidate_matrix()
            candidate_ixes = np.transpose(np.nonzero(self.candidate_matrix))
            candidate_ix = random.choice(candidate_ixes)
            candidate_ix = {'row': candidate_ix[0], 'column': candidate_ix[1]}
            self.candidate_matrix[candidate_ix['row'], candidate_ix['column']] = 0
            center_ix = np.array([candidate_ix['row'], candidate_ix['column']])
            
            for n in range(self.candidate_matrix.shape[0]):
                for m in range(self.candidate_matrix.shape[1]):
                    if self.candidate_matrix[n][m] == 0:
                        continue
                    cell_ix = np.array([n,m])
                    distance = np.linalg.norm(center_ix - cell_ix)
                    if distance < radius:
                        self.candidate_matrix[n][m] = 0
            crop_ix = self.candidate_ix_to_image_ix(candidate_ix)
            image_crop = self.crop_from_current_image(crop_ix)            
            meta = {'positive': True,
                    'count': self.count_sealions_in_crop(crop_ix, skip_pups = True),
                    'column': crop_ix['column'],
                    'row': crop_ix['row'],
                    'id': self.current_image_id,
                    'crop_size': self.crop_size}
            if self.attention:
                image_crop = self.attention_blackout(image_crop,meta)
            self.save_crop(image_crop,meta)
            crops_count += 1
        return crops_count

        
    def find_pos_crops_dataset(self, min_sealions_pos, max_overlap_perc, plot, sliding_perc = 0.1):
        """
        Iterates over all images and extract all the positive crops
        
        :param min_sealions_pos: Int, Minimun amount of sealions required for a crop to be positive
        :param sliding_perc: Float, Percentage of the sliding window size (crop_size) that the sliding
            window moves per slide step
        :param max_overlap_perc: Float, Maximum overlap permitted between positive crops
        """
        self.min_sealions_pos = min_sealions_pos
        self.max_overlap_perc = max_overlap_perc
        self.sliding_perc = sliding_perc
        
        self.sliding_px = round(self.crop_size * self.sliding_perc)
        
        self.plot = plot
        
        image_count = 0
        total = len(list(self.loader.train_original_counts))
        for image_id in self.loader.train_original_counts:
            t0 = time.time()
            self.current_image_id = image_id
            self.current_image = self.loader.load(os.path.join(settings.TRAIN_ORIGINAL_IMAGES_DIR, self.current_image_id+'.jpg'))
            if self.plot:
                plt.figure()
                plt.imshow(self.current_image)
                plt.show()
            self.eval_current_image_size()
            self.init_candidate_matrix()
            logger.info('Searching positive crops in '+self.current_image_id+'.jpg: '+str(self.loader.train_original_counts[self.current_image_id])+', candidate matrix size '+str(self.candidate_matrix.shape))
            self.find_pos_crops_current_image()
            crops_count = self.select_and_save_candidates()
            logger.info('... found and saved '+str(crops_count)+' crops in '+str(time.time() - t0)[:4]+' seconds ('+str(100*image_count/total)[:4]+'% completed)')
            image_count += 1
            if self.plot:
                logger.warning('Execution aborted since plot is activated and this will generate too many plots')
                break
    def random_crop(self):
        """
        Randomly select and returns a crop point on image train_id
        """
        size = ast.literal_eval(self.loader.train_original_counts[self.current_image_id]['size'])
        row = random.randint(0,size[0] - self.crop_size)
        column = random.randint(0,size[1] - self.crop_size)
        crop_ix = {'row':row,'column':column}
        return crop_ix
    
    def attention_blackout(self, image_crop, meta):
        """
        Apply the blackout attention transformation to an image crop
        """
        #TODO
        new_crop = np.zeros(image_crop.shape)
        return new_crop
    
    def save_crop(self, image_crop, meta):
        """
        Saves the crop in the pos or neg folder
        """
        filename = str(meta['count']) + "clions_at" + str(meta['column'])+'-' + str(meta['row']) +"_in"+str(meta['id'])+"_"+str(meta['crop_size'])+"px.jpg"
        if meta['positive']:
            filepath = os.path.join(settings.CROPS_OUTPUT_DIR,'pos',filename)
        else:
            filepath = os.path.join(settings.CROPS_OUTPUT_DIR,'neg',filename)
        scipy.misc.imsave(filepath, image_crop)
            
    def get_blacked_out_perc(self, crop):
        """
        Checks whether the crop contains blacked out regions
        """
        dim1, dim2, _ = crop.shape
        return np.sum(np.sum(crop, 2) == 0) / (dim1*dim2)
    
    def save_negative_crops(self):
        """
        Save negative crops to disk
        """
        # Create weight output dir if it does not exist
        if not os.path.exists(os.path.join(settings.CROPS_OUTPUT_DIR,'neg')):
            os.makedirs(os.path.join(settings.CROPS_OUTPUT_DIR,'neg'))      
        self.negative_crops = sorted(self.negative_crops, key = lambda x: x[1]['id'])
        self.current_image_id = ''
        logger.info("Attempting to write "+str(len(self.negative_crops))+" negative crops to "+os.path.join(settings.CROPS_OUTPUT_DIR,'neg'))
        count = 0
        for crop_meta in self.negative_crops:
            if crop_meta[1]['id'] != self.current_image_id:
                self.current_image_id = crop_meta[1]['id']
                self.current_image = self.loader.load(os.path.join(settings.TRAIN_ORIGINAL_IMAGES_DIR,self.current_image_id+'.jpg'))
                dotted_image = self.loader.load(os.path.join(settings.TRAIN_DOTTED_IMAGES_DIR,self.current_image_id+'.jpg'))
            image_crop = self.crop_from_current_image(crop_meta[0])
            #Quick fix to solve the blackout problem in some images
            temp = self.current_image
            self.current_image = dotted_image
            #print(temp.sum(),self.current_image.sum())
            image_dotted_crop = self.crop_from_current_image(crop_meta[0])
            if self.get_blacked_out_perc(image_dotted_crop) > 0.1:
                logger.info("Blackout crop ignored: "+str(crop_meta[1]))
            else:
                self.save_crop(image_crop,crop_meta[1])
                count += 1
            self.current_image = temp
            if count % 50 == 0:
                logger.info(str(len(self.negative_crops)-count)+" crops left ("+str(100*count/len(self.negative_crops))[:4]+"% completed)")
        logger.info(str(count)+"crops were saved in"+os.path.join(settings.CROPS_OUTPUT_DIR,'neg'))
            

    def find_neg_crops_dataset(self, wanted_crops, max_sealions_neg):
        """
        Find and writes to disk negative herd crops. Each crop is selected by first randomly
        choosing an orignal image, and then randomly cropping inside it.
        
        :param max_sealions_neg: Int, Maximum amount of sealions required for a crop to be negative
        :wanted_crops: Int, desired amount of crops to be generated
        """
        self.max_sealions_neg = max_sealions_neg
        self.negative_crops = []
        train_ids = list(self.loader.train_original_counts.keys())
        
        
        while len(self.negative_crops) < wanted_crops:
            self.current_image_id = random.choice(train_ids)
            if self.current_image_id in self.loader.train_original_mismatched:
                # Skip images marked as mismatched
                continue
            crop_ix = self.random_crop()
            n_sealions = self.count_sealions_in_crop(crop_ix, skip_pups = False)
            if n_sealions > self.max_sealions_neg:
                #print(n_sealions)
                continue
            #self.current_image = self.loader.load(os.path.join(settings.TRAIN_ORIGINAL_IMAGES_DIR, self.current_image_id+'.jpg'))
            #self.plot_crop_ix(crop_ix)
            #print(n_sealions,self.current_image.shape)

            meta = {'positive': False,
                    'count': n_sealions,
                    'id': self.current_image_id,
                    'column': crop_ix['column'],
                    'row': crop_ix['row'],
                    'crop_size': self.crop_size}
            self.negative_crops.append((crop_ix,meta))
        self.save_negative_crops()



def crop(img, bounding_boxes, out_size = (300, 300), zoom_factor = 0.7):
    """
    Crop an image using the bounding box(es) to one or more cropped image(s).

    :param img: The image to generate the crops for
    :param bounding_boxes: A list of dictionaries of the form x, y, width, height, and optionally class.
    :param out_size: The size of the output crops
    :param zoom_factor: The factor with which to zoom in to the crop (relative to the fish size)
                        == 1: no zoom
                        >  1: zoom in
                        <  1: zoom out
    :return: A list of crops (and potentially a list of classes for each crop, if classes were given in the input)
    """

    crops = []

    for bounding_box in bounding_boxes:
        box_x_low, box_x_high, box_y_low, box_y_high = zoom_box(bounding_box, img.shape, zoom_factor)

        # Crop
        crop = img[box_y_low : box_y_high, box_x_low : box_x_high, :]

        # Resize to output size
        crop = scipy.misc.imresize(crop, size = out_size)

        # Add to crops list
        if "class" in bounding_box:
            crops.append((crop, bounding_box["class"]))
        else:
            crops.append(crop)

    return zip(*crops)

def generate_individual_crops(num_negative_crops:int):
    """
    """
    loader = data.Loader()
    images = loader.load_original_images()
    positive_output_dir = os.path.join(settings.CROPS_OUTPUT_DIR, "positive")
    negative_output_dir = os.path.join(settings.CROPS_OUTPUT_DIR, "negative")

    num_images = len(images)
    num_neg_crops_per_image = int(num_negative_crops / num_images)
    num_neg_crops_remainder = num_neg_crops_per_image % num_images

    n = 0
    for image in images:
        n += 1

        num_neg_crops_this_image = (num_neg_crops_per_image + 1) if n <= num_neg_crops_remainder else num_neg_crops_per_image
            
        img = image['x']()
        m = image['m']

        # bounding_boxes = []
        # for coordinate in m['coordinates']:
        #     bounding_boxes.append(...)
        # positive_crops = crop(img, bounding_boxes, out_size = (..., ...), zoom_factor = 1)
        #
        # ... negative crops

        pass
