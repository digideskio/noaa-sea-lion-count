import data
import numpy as np
import os
import scipy
import settings
import vigra

DEFAULT_SCALES = [0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0] # in pixels

def preproc(img):
    return img/128 - 1

def laplacian_of_gaussian(img, scale):
    return vigra.filters.laplacianOfGaussian(img, scale=scale)

def gaussian_gradient_magnitude(img, scale):
    return vigra.filters.gaussianGradientMagnitude(img, sigma=scale)

def gaussian_smoothing(img, scale):
    return vigra.filters.gaussianSmoothing(img, sigma=scale)

def difference_of_gaussians(img, scale):
    return vigra.filters.gaussianSmoothing(img, sigma=scale) - vigra.filters.gaussianSmoothing(img, sigma=0.66*scale)

def gaussian_smoothing_and_difference_of_gaussians(img, scale):
    # combines the above two functions, which share a common function call
    gs = vigra.filters.gaussianSmoothing(img, sigma=scale)
    dog = gs - vigra.filters.gaussianSmoothing(img, sigma=0.66*scale)
    return gs, dog

def hessian_of_gaussian_eigenvalues(img, scale):
    return vigra.filters.hessianOfGaussianEigenvalues(np.sum(img, axis=2), scale=scale)
    
def structure_tensor_eigenvalues(img, scale):
    return vigra.filters.structureTensorEigenvalues(img, innerScale=scale, outerScale=0.5*scale)

def generate_features(img, base_out_path, scales=DEFAULT_SCALES):
    # base_out_path is "path/to/output/dir/imageid"
    img = preproc(img)
    
    for scale in scales:
        settings.logger.info('Generating scale = ' + ('%g' % scale) + ' features...')
        outpath = base_out_path + ('_%g_' % scale)
        out = laplacian_of_gaussian(img, scale)
        out.writeImage(outpath + 'log.jpg')
        
        out = gaussian_gradient_magnitude(img, scale)
        out.writeImage(outpath + 'ggm.jpg')
        
        out, out2 = gaussian_smoothing_and_difference_of_gaussians(img, scale)
        out.writeImage(outpath + 'gs.jpg')
        out2.writeImage(outpath + 'dog.jpg')
        
        out = hessian_of_gaussian_eigenvalues(img, scale)
        (out[:,:,0]).writeImage(outpath + 'hoge1.jpg')
        (out[:,:,1]).writeImage(outpath + 'hoge2.jpg')
        
        out = structure_tensor_eigenvalues(img, scale)
        (out[:,:,0]).writeImage(outpath + 'ste1.jpg')
        (out[:,:,1]).writeImage(outpath + 'ste2.jpg')

def run_feature_generation(dataset, start=0, end=-1):
    loader = data.Loader()
    images = loader.load_original_images(dataset=dataset)
    
    if dataset == 'train':
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
        settings.logger.info('Generating features for image %d (%s.jpg)...' % (idx, imageid))
        
        image = vigra.readImage(os.path.join(settings.TRAIN_ORIGINAL_IMAGES_DIR, imageid + '.jpg'))
        generate_features(image, os.path.join(outdir, imageid))


