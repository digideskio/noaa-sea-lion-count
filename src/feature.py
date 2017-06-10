from cropping import coords_overlap
import data
import numpy as np
import os
import random
import scipy
import settings
import vigra

DEFAULT_SCALES = [0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0] # in pixels
DEFAULT_NUM_NEGATIVE_CROPS = 30
DEFAULT_SEA_LION_SIZE = 150
DEFAULT_WINDOW_SLIDE = 15
DEFAULT_CROP_REGION_SIZE = 400
DEFAULT_MAX_POS_OVERLAP = 0.5


def preproc(img):
    return img/128 - 1

def laplacian_of_gaussian(img, scale, roi=None):
    return vigra.filters.laplacianOfGaussian(img, scale=scale, roi=roi)

def gaussian_gradient_magnitude(img, scale, roi=None):
    return vigra.filters.gaussianGradientMagnitude(img, sigma=scale, roi=roi)

def gaussian_smoothing(img, scal, roi=None):
    return vigra.filters.gaussianSmoothing(img, sigma=scale, roi=roi)

def difference_of_gaussians(img, scale, roi=None):
    return vigra.filters.gaussianSmoothing(img, sigma=scale, roi=roi) - vigra.filters.gaussianSmoothing(img, sigma=0.66*scale, roi=roi)

def gaussian_smoothing_and_difference_of_gaussians(img, scale, roi=None):
    # combines the above two functions, which share a common function call
    gs = vigra.filters.gaussianSmoothing(img, sigma=scale, roi=roi)
    dog = gs - vigra.filters.gaussianSmoothing(img, sigma=0.66*scale, roi=roi)
    return gs, dog

def hessian_of_gaussian_eigenvalues(img, scale, roi=None):
    return vigra.filters.hessianOfGaussianEigenvalues(np.sum(img, axis=2), scale=scale, roi=roi)
    
def structure_tensor_eigenvalues(img, scale, roi=None):
    return vigra.filters.structureTensorEigenvalues(img, innerScale=scale, outerScale=0.5*scale, roi=roi)

def generate_features(img, base_out_path, scales=DEFAULT_SCALES, patches=None, crop_size=DEFAULT_CROP_REGION_SIZE):
    # base_out_path is "path/to/output/dir/imageid"
    img = preproc(img)
    
    if patches:
        num_crops = len(patches)
        patch_roi = lambda coord: ((coord[0], coord[1]), (coord[0]+crop_size, coord[1]+crop_size))
        settings.logger.info('Generating %d crops for this image...' % num_crops)
    else:
        patch_roi = lambda _: None
        patches = [(-1,-1)]
    
    for i, patch in enumerate(patches):
        roi = patch_roi(patch)
        
        if roi:
            outpath = base_out_path + ('_%d-%d-%d-%d_' % (patch[0], patch[1], crop_size, crop_size))
            settings.logger.info('Generating features for crop %d/%d' % (i+1, num_crops))
        else:
            outpath = base_out_path + '_'
        
        for scale in scales:
            settings.logger.info('Generating scale = ' + ('%g' % scale) + ' features...')
            
            out = laplacian_of_gaussian(img, scale, roi=roi)
            out.writeImage(outpath + ('log-%g.png' % scale))
            
            out = gaussian_gradient_magnitude(img, scale, roi=roi)
            out.writeImage(outpath + ('ggm-%g.png' % scale))
            
            out, out2 = gaussian_smoothing_and_difference_of_gaussians(img, scale, roi=roi)
            out.writeImage(outpath + ('gs-%g.png' % scale))
            out2.writeImage(outpath + ('dog-%g.png' % scale))
            
            out = hessian_of_gaussian_eigenvalues(img, scale, roi=roi)
            (out[:,:,0]).writeImage(outpath + ('hoge1-%g.png' % scale))
            (out[:,:,1]).writeImage(outpath + ('hoge2-%g.png' % scale))
            
            out = structure_tensor_eigenvalues(img, scale, roi=roi)
            (out[:,:,0]).writeImage(outpath + ('ste1-%g.png' % scale))
            (out[:,:,1]).writeImage(outpath + ('ste2-%g.png' % scale))

def sliding_window_crop_generation(sea_lion_coordinates, image_size, ignore_pups=False,
                                   num_negative_crops=DEFAULT_NUM_NEGATIVE_CROPS, crop_size=DEFAULT_CROP_REGION_SIZE,
                                   max_overlap=DEFAULT_MAX_POS_OVERLAP, sea_lion_size=DEFAULT_SEA_LION_SIZE, window_slide=DEFAULT_WINDOW_SLIDE):

    imwidth, imheight, _ = image_size
    bases_x = range(0, imwidth  - crop_size, window_slide)
    bases_y = range(0, imheight - crop_size, window_slide)
    
    dot_within_region = lambda dot, region_bound, region_size: dot > region_bound and dot < region_bound + region_size # dot is in the centre, so if the centre is in this region, at least half the sea lion is in as well
    coord_in_bbox = lambda coord, base_pos: dot_within_region(coord[0], base_pos[0], crop_size) and dot_within_region(coord[1], base_pos[1], crop_size)
    
    # Generate negative crops
    crops = []
    while len(crops) < num_negative_crops:
        crop_coord = (random.randrange(imwidth  - crop_size), random.randrange(imheight - crop_size))
        
        if not any(coord_in_bbox(coord, crop_coord) for coord, category in sea_lion_coordinates): # ensure negative
            crops.append(crop_coord)
    
    # Construct positive/negative matrix
    pos_neg_matrix = np.empty((len(bases_x), len(bases_y)), dtype=np.bool)
    for i, base_x in enumerate(bases_x):
        for j, base_y in enumerate(bases_y):
            if any(coord_in_bbox(coord, (base_x, base_y)) for coord, category in sea_lion_coordinates if not (ignore_pups and category == 'pups')):
                pos_neg_matrix[i,j] = True
            else:
                pos_neg_matrix[i,j] = False
    
    # Generate positive crops
    while np.sum(pos_neg_matrix) > 0:
        pos_is, pos_js = np.where(pos_neg_matrix)
        pos_i = random.randrange(len(pos_is))
        
        crop_i = pos_is[pos_i]
        crop_j = pos_js[pos_i]
        
        crop_x = bases_x[crop_i]
        crop_y = bases_y[crop_j]
        crops.append((crop_x, crop_y))
        
        for other_i, other_x in enumerate(bases_x):
            for other_j, other_y in enumerate(bases_y):
                if coords_overlap((crop_x, crop_y), (other_x, other_y), crop_size) >= max_overlap:
                    pos_neg_matrix[other_i, other_j] = False
    
    return crops
        

def run_feature_generation(dataset, start=0, end=-1, patches=False, ignore_pups=False):
    loader = data.Loader()
    images = loader.load_original_images(dataset=dataset)
    
    if dataset == 'train':
        if patches:
            outdir = settings.DENSITY_MAP_FEATURE_CROPS_DIR
        else:
            outdir = settings.TRAIN_FEATURES_DIR
    elif dataset == 'test_st1':
        outdir = settings.TEST_FEATURES_DIR
    else:
        raise Exception('Data set not implemented: ' + dataset)
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    if end == -1:
        end = len(images)
    
    for idx in range(start, end):
        imageid = images[idx]['m']['filename']
        settings.logger.info('Generating crops image %d (%s.jpg)...' % (idx, imageid))
        
        image = vigra.readImage(os.path.join(settings.TRAIN_ORIGINAL_IMAGES_DIR, imageid + '.jpg'))
        
        if patches:
            sea_lions = [((round(float(coord['x_coord'])), round(float(coord['y_coord']))), coord['category']) for coord in images[idx]['m']['coordinates']]
            crops = sliding_window_crop_generation(sea_lions, image.shape, ignore_pups)
            
            generate_features(image, os.path.join(outdir, imageid), patches=crops)
        else:
            generate_features(image, os.path.join(outdir, imageid))


