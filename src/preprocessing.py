import collections
import numpy as np
import random
import scipy.linalg
import scipy.misc
import skimage.color
import skimage.util

# From scikit image github
yuv_from_rgb = np.array([[ 0.299     ,  0.587     ,  0.114      ],
                         [-0.14714119, -0.28886916,  0.43601035 ],
                         [ 0.61497538, -0.51496512, -0.10001026 ]])
rgb_from_yuv = scipy.linalg.inv(yuv_from_rgb)

def clamp(arr):
    arr[arr < 0] = 0
    arr[arr > 1] = 1
    return arr

def rgb2yuv(arr):
    arr = skimage.util.dtype.img_as_float(arr)
    return clamp(np.dot(arr, yuv_from_rgb.T.copy()))

def yuv2rgb(arr):
    arr = skimage.util.dtype.img_as_float(arr)
    return clamp(np.dot(arr, rgb_from_yuv.T.copy()))

def random_negative_boxes(img, positives, num_fps):
    mean_bbox_width = 200
    mean_bbox_height = 150
    neg_overlap_ratio = 0.10
    
    shape = img.shape
    
    intersection_bbox = lambda cand, fish: max(0, min(cand['x']+cand['width'], fish['x']+fish['width']) - max(cand['x'], fish['x'])) * max(0, min(cand['y']+cand['height'], fish['y']+fish['height']) - max(cand['y'], fish['y']))
    containment_ratio = lambda cand, fish: intersection_bbox(cand, fish) / float(fish['width']*fish['height'])
    
    crops = []
    while len(crops) < num_fps:
        width = min(mean_bbox_width*random.lognormvariate(0, 0.75), img.shape[1]/2.0)
        height = min(mean_bbox_height*random.lognormvariate(0, 0.5), img.shape[0]/2.0)
        x = (shape[1]-width) * random.random()
        y = (shape[0]-height) * random.random()
        crop = {'x': x, 'y': y, 'width': width, 'height': height, 'class': 'NoF'}
        
        if not any(containment_ratio(zoom_box(crop, shape, output_dict=True), bbox) > neg_overlap_ratio for bbox in positives): # negative even when zoomed out
            crops.append(crop)
    
    return crops
    

def zoom_box(bounding_box, img_shape, zoom_factor = 0.7, output_dict=False):
    (size_y, size_x, channels) = img_shape
    
    zoom_out_factor = 1 / zoom_factor
    
    box_x = bounding_box['x']
    box_y = bounding_box['y']
    box_w = bounding_box['width']
    box_h = bounding_box['height']

    # Make square by creating seperate zoom-out factors for each axis
    if box_w > box_h:
        zoom_out_factor_x = zoom_out_factor
        zoom_out_factor_y = zoom_out_factor * (box_w / box_h)
    else: 
        zoom_out_factor_x = zoom_out_factor * (box_h / box_w)
        zoom_out_factor_y = zoom_out_factor

    # Zoom out (or in)
    box_x = box_x + (box_w - zoom_out_factor_x * box_w) / 2;
    box_y = box_y + (box_h - zoom_out_factor_y * box_h) / 2;
    box_w *= zoom_out_factor_x;
    box_h *= zoom_out_factor_y;

    # To integer values
    box_x = round(box_x)
    box_y = round(box_y)
    box_w = round(box_w)
    box_h = round(box_h)

    # Ensure the bounding box is not greater than the image itself
    if box_w > size_x:
        box_w = size_x

    if box_h > size_y:
        box_h = size_y

    # Translate (and potentially shrink) box to ensure it falls within the image region
    if box_x < 0:
        box_x = 0
    elif box_x + box_w > size_x:
        box_x = size_x - box_w

    if box_y < 0:
        box_y = 0
    elif box_y + box_h > size_y:
        box_y = size_y - box_h
    
    if output_dict:
        return {'x': box_x, 'width': box_w, 'y': box_y, 'height': box_h}
    else:
        return box_x, box_x + box_w, box_y, box_y + box_h

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
