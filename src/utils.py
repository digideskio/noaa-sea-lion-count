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
    return image[y_coordinate : y_coordinate + crop_size, x_coordinate : x_coordinate + crop_size, :]