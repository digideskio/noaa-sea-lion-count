import data
import numpy as np
import os
import scipy
import settings
from skimage import img_as_float
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
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
    outimg = np.empty_like(img)
    outimg2 = np.empty_like(img)
    for d in range(3):
        Hxx, Hxy, Hyy = hessian_matrix(img[:,:,d], sigma=scale, order='rc')
        outimg[:,:,d], outimg2[:,:,d] = hessian_matrix_eigvals(Hxx, Hxy, Hyy)
    return outimg, outimg2
    
def structure_tensor_eigenvalues(img, scale):
    outimg = vigra.filters.structureTensorEigenvalues(img, innerScale=scale, outerScale=0.5*scale)
    return outimg

def generate_features(img, base_out_path, scales=DEFAULT_SCALES):
    # base_out_path is "path/to/output/dir/imageid"
    img = preproc(img)
    imgt = np.transpose(img, (1,0,2)) # HoGE is from scipy which does not respect axistags
    
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
        
        out, out2 = hessian_of_gaussian_eigenvalues(imgt, scale)
        scipy.misc.imsave(outpath + 'hoge1.jpg', out)
        scipy.misc.imsave(outpath + 'hoge2.jpg', out2)
        
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
        settings.logger.info('Generating features for image ' + str(idx) + '...')
        
        imageid = images[idx]['m']['filename']
        image = vigra.readImage(os.path.join(settings.TRAIN_ORIGINAL_IMAGES_DIR, imageid + '.jpg'))
        
        generate_features(image, os.path.join(outdir, imageid))


