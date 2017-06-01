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
