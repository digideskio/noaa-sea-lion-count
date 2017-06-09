import numpy as np
import scipy
from skimage import img_as_float
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from datetime import datetime
import vigra

def preproc(img):
    return 2*img_as_float(img) - 1

def laplacian_of_gaussian(img, scale):
    return vigra.filters.laplacianOfGaussian(img, scale=scale)

def gaussian_gradient_magnitude(img, scale):
    return vigra.filters.gaussianGradientMagnitude(img, sigma=scale)

def gaussian_smoothing(img, scale):
    return vigra.filters.gaussianSmoothing(vimg, sigma=scale)

def difference_of_gaussians(img, scale):
    return vigra.filters.gaussianSmoothing(img, sigma=scale) - vigra.filters.gaussianSmoothing(img, sigma=0.66*scale)

def gaussian_smoothing_and_difference_of_gaussians(img, scale):
    # combines the above two functions, which share a common function call
    gs = vigra.filters.gaussianSmoothing(img, sigma=scale)
    dog = gs - vigra.filters.gaussianSmoothing(img, sigma=0.66*scale)
    return gs, dog

def hessian_of_gaussian_eigenvalues(img, scale):
    outimg = np.empty_like(vimg)
    outimg2 = np.empty_like(vimg)
    for d in range(3):
        Hxx, Hxy, Hyy = hessian_matrix(vimg[:,:,d], sigma=scale, order='rc')
        outimg[:,:,d], outimg2[:,:,d] = hessian_matrix_eigvals(Hxx, Hxy, Hyy)
    return np.transpose(outimg, (1,0,2))), np.transpose(outimg2, (1,0,2)))
    
def structure_tensor_eigenvalues(img, scale):
    outimg = vigra.filters.structureTensorEigenvalues(vimg, innerScale=scale, outerScale=0.5*scale)
    return outimg[:,:,0], outimg[:,:,1]

def generate_features(img, scales, base_img_path):
    # base_img_path is "path/to/output/dir/imageid"
    img = preproc(img)
    for scale in scales:
        out = laplacian_of_gaussian(img, scale)
        out.writeImage(base_img_path + '_log.jpg')
        
        out = gaussian_gradient_magnitude(img, scale)
        out.writeImage(base_img_path + '_ggm.jpg')
        
        out, out2 = gaussian_smoothing_and_difference_of_gaussians(img, scale)
        out.writeImage(base_img_path + '_gs.jpg')
        out2.writeImage(base_img_path + '_dog.jpg')
        
        out, out2 = hessian_of_gaussian_eigenvalues(img, scale)
        scipy.misc.imsave(base_img_path + '_hoge1.jpg', np.transpose(out, (1,0,2)))
        scipy.misc.imsave(base_img_path + '_hoge2.jpg', np.transpose(out2, (1,0,2)))
        
        out, out2 = structure_tensor_eigenvalues(img, scale)
        out.writeImage(base_img_path + '_ste1.jpg')
        out2.writeImage(base_img_path + '_ste2.jpg')


