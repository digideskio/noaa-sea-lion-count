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

def blacken_crop(crop, rad):
    """
    Return a crop with only the sealion visible and the corners turned black.

    :params crop: The image returned by crop_image
    :params rad: The radius of the output circle.
    """
    from PIL import Image
    from PIL import ImageDraw
    crop = Image.fromarray(crop.astype('uint8'))
    circle = Image.new('L', (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2, rad * 2), fill=255)
    alpha = Image.new('L', crop.size, 255)
    w, h = crop.size
    alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
    alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
    alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
    crop.putalpha(alpha)
    return crop
