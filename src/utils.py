"""
Module containing some utilities
"""

import os

def remove_key_from_dict(dict, *keys):
    """
    Remove keys from a dictionary, and return the new dictionary.

    :param dict: The dictionary to remove a key from
    :param keys: Keys to remove from the dictionary
    :return: A dictionary with the key removed
    """
    d = dict.copy()
    
    for key in keys:
        if key in d:
            del d[key]
    return d

def get_gaussian_mark(sigma):
    """
    Returns a numpy 2D with shape (60, 60) with a gaussian kernel normalized
    """
    import numpy as np
    size_aux = 200.
    radius = 100
    x = np.linspace(-10, 10, size_aux)
    y = np.linspace(-10, 10, size_aux)
    x, y = np.meshgrid(x, y)
    z = (1/(2*np.pi*sigma**2) * np.exp(-(x**2/(2*sigma**2)
         + y**2/(2*sigma**2))))
    mark = z[round(size_aux/2-radius):round(size_aux/2+radius),round(size_aux/2-radius):round(size_aux/2+radius)]
    mark /= mark.max()
    return mark

def get_multivariate_normal_pdf(x = [[-5, 5], [-10, 10]], output_resolution = [30, 30], cov = [[1.0, 0], [0, 1.0]]):
    """
    Create a multivariate normal density.

    :param x: The min and max x-values to get the density values for.
    :param output_resolution: The number of output points the axis should have.
    :param cov: The covariance matrix.
    :return: The multivariate normal density with shape <output_resolution>
             on domain <x> with mean 0 and covariance matrix <cov>.
    """
    import numpy as np
    from scipy.stats import multivariate_normal

    mean = [0] * len(cov)

    var = multivariate_normal(mean=mean, cov=cov)

    axes = []
    for x_, diameter in zip(x, output_resolution):
        axes.append(np.linspace(x_[0], x_[1], diameter))

    mesh = np.meshgrid(*axes)
    pos = np.empty(mesh[0].shape + (len(mesh),))
    for i in range(len(mesh)):
        pos[:, :, i] = mesh[i] 

    return multivariate_normal.pdf(pos, mean, cov)

def get_file_name_part(full_path):
    """
    Get the name of the file (without extension) given by the specified full path.

    E.g. "/path/to/file.ext" becomes "file"

    :param full_path: The full path to the file
    :return: The name of the file without the extension (and without the path)
    """
    base = os.path.basename(full_path)
    return os.path.splitext(base)[0]

def crop_image(image, coordinates, crop_size):    
    """
    Returns a square shaped crop from an image.
    
    :param image: The image from which to take the crop
    :param coordinates: Tuple that contains the left upper corner coordinates of the crop
    :param crop_size: The size of the desired crop
    """
    x_coordinate, y_coordinate = coordinates[0], coordinates[1]
    if len(image.shape) == 3: # RGB image
        return image[y_coordinate : y_coordinate + crop_size, x_coordinate : x_coordinate + crop_size, :]
    else: # regular 2D matrix
        return image[y_coordinate : y_coordinate + crop_size, x_coordinate : x_coordinate + crop_size]

def blackout(crop, locations, sea_lion_size):
    """
    Draws circles of given size at a given list of locations in an image

    :param crop: The crop to work on
    :param locations: A list of locations of top-left corners of circles
    :param diameter: The diameter of the circles to be drawn
    """
    import numpy as np
    from PIL import Image, ImageDraw
    
    height, width, _ = crop.shape
    crop = Image.fromarray(crop.astype('uint8'))
    mask = Image.new('1', (width, height), color=0)
    draw = ImageDraw.Draw(mask)
    
    for x, y in locations:
        draw.ellipse((x, y, x + sea_lion_size, y + sea_lion_size), fill=255)
    
    result = Image.new('RGB', (width, height), color=(0,0,0))
    result.paste(crop, mask=mask)
    del crop, mask, draw
    
    return np.array(result)
