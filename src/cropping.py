"""
Module for data cropping functionality.
"""

import os
import random
import scipy

import data
import settings
import utils

logger = settings.logger.getChild('cropping')

class RegionCropper:
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
        logger.debug("Cropper intialisated with:\n\t crop_size: %s\n\ttotal_crops: %s\n\tpos_perc: %s" % (self.crop_size, total_crops, pos_perc))
        
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

def coords_overlap(coord1, coord2, bbox_size):
    xmin1, ymin1 = coord1
    xmin2, ymin2 = coord2
    
    xmax1, ymax1 = xmin1+bbox_size, ymin1+bbox_size
    xmax2, ymax2 = xmin2+bbox_size, ymin2+bbox_size
    
    intersection = max(0, min(xmax1, xmax2) - max(xmin1, xmin2)) * max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
    return intersection / float(bbox_size*bbox_size)
    

def generate_individual_crops(sea_lion_size, num_negative_crops, ignore_pups=False):
    """
    :param sea_lion_size: The width/height (in the actual image) of a sea lion crop (default 100 by 100)
    :param num_negative_crops: The total number of negative crops generated, over all images
    :param ignore_pups: If true, pups will not result in positive crops
    """
    loader = data.Loader()
    images = loader.load_original_images()
    coordsdict = loader.get_train_original_coordinates()
    
    dir = settings.CROPS_OUTPUT_DIR
    os.makedirs(dir)

    num_images = len(images)
    num_neg_crops_per_image = int(num_negative_crops / num_images)
    num_neg_crops_remainder = num_neg_crops_per_image % num_images
    
    n = 0
    for image in images:
        n += 1
        logger.info('Cropping image ' + str(n) + '...')

        num_neg_crops_this_image = (num_neg_crops_per_image + 1) if n <= num_neg_crops_remainder else num_neg_crops_per_image
            
        img = image['x']()
        filename = image['m']['filename']
        img_height, img_width, _ = img.shape
        coords = [(max(0, min(img_width-sea_lion_size, int(float(coord['x_coord']) - sea_lion_size/2))),
                   max(0, min(img_height - sea_lion_size, int(float(coord['y_coord']) - sea_lion_size/2))),
                   coord['category']) for coord in coordsdict[filename]]
        
        num_neg = 0
        negcoords = []
        while num_neg < num_neg_crops_this_image:
            x_coord = random.randint(0, img_width  - sea_lion_size)
            y_coord = random.randint(0, img_height - sea_lion_size)
            
            # Check for overlap with any sea lions (also slightly outside of the bounding box, to avoid matching bigger sea lions' parts)
            if any(coords_overlap((sea_lion[0], sea_lion[1]), (x_coord, y_coord), 1.25*sea_lion_size) > 0 for sea_lion in coords):
                continue
            
            negcoords.append((x_coord, y_coord, 'negative'))
            num_neg += 1

        for x_coord, y_coord, category in negcoords + coords:
            if ignore_pups and coord['category'] == 'pups':
                continue
            
            # Crop
            crop_img = utils.crop_image(img, (x_coord, y_coord), sea_lion_size)
            #print((x_coord, y_coord, img.shape, crop_img.shape))

            # Resize to output size - not needed for networks if it's uniform
            #crop = scipy.misc.imresize(crop, size = out_size)

            cropname = category + '_id' + filename[:-4] + '_' + str(1 * (num_neg<=0)) + 'clions_at_' + str(x_coord) + '-' + str(y_coord) + '_' + str(sea_lion_size) + 'px.jpg'
            num_neg -= 1
            
            scipy.misc.imsave(os.path.join(dir, cropname), crop_img)
            
    
