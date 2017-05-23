"""
Module for data cropping functionality.
"""

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