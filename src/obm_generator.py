    

import os
import time
import sys
import network
import utils
from matplotlib import pyplot as plt
import scipy
import glob
import settings
import data
from scipy.misc import imresize, imread
import cv2
import numpy as np
from multiprocessing import Pool
#%matplotlib inline
from itertools import chain, islice,groupby, count
import utils

    
def generate_obms(impaths): 
    data_type = 'original_test'
    #data_type = 'region_crops'
    prediction_class_type = 'odm'
    #prediction_class_type = 'single'
    validate = False
    class_balancing = False
    input_shape = (224, 224, 3)
    batch_size = 1
    crop_size = 400
    arch = 'xception'
    input_weights_name = 'xception-lay106-heatmap_crops-ep011-tloss0.0068-vloss0.0067.hdf5'
    tl = network.TransferLearningSeaLionHeatmap(data_type = data_type, input_shape = input_shape, prediction_class_type = prediction_class_type, class_balancing= class_balancing, mini_batch_size=batch_size, validate = validate)
    #tl = network.TransferLearningSeaLionOrNoSeaLion(data_type = data_type, input_shape = input_shape, prediction_class_type = prediction_class_type, class_balancing= class_balancing, mini_batch_size=mini_batch_size, validate = validate)
    tl.build(arch, input_shape = input_shape)
    tl.load_weights(input_weights_name)

    cnn_output_shape = tl.model.layers[-1].output_shape[1:-1]

    for impath in list(impaths):
        meta = {'filename':utils.get_file_name_part(impath)}
        if os.path.isfile(os.path.join(os.path.join(settings.OBMS_OUTPUT_DIR,meta['filename']+'_obm.h5.npy'))):
            print("skipping")
            continue
        test_image_original = scipy.misc.imread(impath)
        t0 = time.time()
        #test_image_original, meta = iterator.__next__()
        #test_image_original = test_image_original[0]
        #meta = meta[0]
        test_image_original = test_image_original / test_image_original.max()
        aux_height = test_image_original.shape[0] - test_image_original.shape[0] % crop_size + int(1.5 * crop_size)
        aux_width = test_image_original.shape[1] - test_image_original.shape[1] % crop_size + int(1.5 * crop_size)
        padded = np.zeros((aux_height,aux_width,3))
        padded[:test_image_original.shape[0], :test_image_original.shape[1], :] = test_image_original
        test_image_1 = padded[:aux_height-int(0.5*crop_size),:aux_width-int(0.5*crop_size)]
        test_image_2 = padded[int(0.5*crop_size):,int(0.5*crop_size):]
        #print(test_image.mean())
        obms = []
        plot = 0
        nrows = int(test_image_1.shape[0]/crop_size)
        ncolumns = int(test_image_1.shape[1]/crop_size)
        total = float((nrows * ncolumns)) * 2
        for wa in [test_image_original,padded,test_image_1,test_image_2]:
            if plot:
                plt.figure()
                plt.imshow(wa)
                plt.show()
        settings.logger.info("Going for image "+str(meta['filename']))#+" with shape "+str(test_image_original.shape),", padded ",str(padded.shape))
        count = 0.0
        for test_image in [test_image_1, test_image_2]: 
            if plot:
                plt.figure()
                plt.imshow(test_image)
                plt.show()
            full_obm = []
            for row in range(nrows):
                row_obms = []
                for column in range(ncolumns):
                    if count%15==0:
                        settings.logger.info(str(100*count/total)[:4]+"% completed of "+meta['filename'])
            
                    crop = utils.crop_image(test_image, (column*crop_size, row*crop_size), crop_size)
                    if utils.get_blacked_out_perc(crop)>0.85:
                        obm = np.zeros(cnn_output_shape)
                    else:
                        crop = scipy.misc.imresize(crop, input_shape)
                        if crop.max() > 0:
                            crop = crop / crop.max()
                
                        crop = np.expand_dims(crop, axis = 0)
                        obm = tl.model.predict(crop)
                        obm = np.squeeze(obm)
                    row_obms.append(obm)
                    count += 1
                row_obms = np.hstack(row_obms)
                full_obm.append(row_obms)
            
            full_obm = np.vstack(full_obm)
            #print(full_obm.shape, full_obm.max(),full_obm.mean(),full_obm.min())
            if plot:
                plt.figure()
                plt.imshow(np.squeeze(full_obm), cmap = 'gray')
                plt.title(str(full_obm.sum()))
                plt.show()
            obms.append(full_obm)
        final_obm_1 = np.zeros((obms[0].shape[0]+int(cnn_output_shape[0]/2), obms[0].shape[1]+int(cnn_output_shape[1]/2)))
        final_obm_2 = final_obm_1.copy()
        
        final_obm_1[:obms[0].shape[0], :obms[0].shape[1]] = obms[0]
        final_obm_2[int(cnn_output_shape[0]/2):, int(cnn_output_shape[0]/2):] = obms[1]
        
        final_obm = (final_obm_2 + final_obm_1)/2
        
        
        
        full_obm = final_obm
        trunc_img = padded#[:crop_size*nrows,:crop_size*ncolumns]
        trunc_img = scipy.misc.imresize(trunc_img, (full_obm.shape[0],full_obm.shape[1],3))
        trunc_img = trunc_img / trunc_img.max()
        red_obm = np.zeros((full_obm.shape[0],full_obm.shape[1],3))
        red_obm[:,:,0] = full_obm
        red_obm = red_obm / red_obm.max()
        obms.append(red_obm)
        img_sum = cv2.addWeighted(src1 = trunc_img, alpha = 1, src2 = red_obm, beta = 0.6, gamma = 0.001)
        img_sum = img_sum / img_sum.max()
        scipy.misc.imsave('delete/'+meta['filename']+'_obm.jpg',img_sum)
        settings.logger.info(meta['filename']+" completed in "+str(time.time()-t0)+" seconds")
        if plot:
            plt.figure()
            plt.imshow(img_sum)
            plt.show()
        np.save(os.path.join(settings.OBMS_OUTPUT_DIR,meta['filename']+'_obm.h5'), final_obm)

def generate_obms2(impaths): 
    data_type = 'original_test'
    #data_type = 'region_crops'
    prediction_class_type = 'odm'
    #prediction_class_type = 'single'
    validate = False
    class_balancing = False
    input_shape = (224, 224, 3)
    batch_size = 1
    crop_size = 400
    arch = 'xception'
    input_weights_name = 'xception-lay106-heatmap_crops-ep011-tloss0.0068-vloss0.0067.hdf5'
    tl = network.TransferLearningSeaLionHeatmap(data_type = data_type, input_shape = input_shape, prediction_class_type = prediction_class_type, class_balancing= class_balancing, mini_batch_size=batch_size, validate = validate)
    #tl = network.TransferLearningSeaLionOrNoSeaLion(data_type = data_type, input_shape = input_shape, prediction_class_type = prediction_class_type, class_balancing= class_balancing, mini_batch_size=mini_batch_size, validate = validate)
    tl.build(arch, input_shape = input_shape)
    tl.load_weights(input_weights_name)

    cnn_output_shape = tl.model.layers[-1].output_shape[1:-1]

    for impath in list(impaths):
        im = scipy.misc.imread(impath)
        im = im / im.max()
        im = np.expand_dims(im, axis = 0)
        obm =tl.model.predict(im)
        filename = utils.get_file_name_part(impath)
        print(im.shape,im.mean(),obm.shape,obm.mean(),filename)
        nclions = int(filename.split('clions')[0])
        np.save(os.path.join(settings.OBMS_OUTPUT_DIR,'train',filename+'_obm_train'), obm)
        
def generate_obms_fast():
    cpus = 30
    def chunks(l, n):
        """Yield successive n-sized chunks from lr."""
        chunks = []
        chunk_size = int(len(l)/n)
        for i in range(n+1):
            chunks.append(l[i*chunk_size:(i+1)*chunk_size])
        return chunks
    
    impaths = sorted(glob.glob(os.path.join(settings.TEST_DIR,'original','*')))
    print(len(impaths))
    impaths = np.array(impaths)
    impaths = chunks(impaths, cpus)
    print(len(impaths))
    for a in impaths:
        print(len(a))
        print(str(a)[:500])
    
    with Pool(cpus) as pool:
        pool.starmap(generate_obms,zip(impaths))
        
def generate_obms_fast2():
    cpus = 20
    def chunks(l, n):
        """Yield successive n-sized chunks from lr."""
        chunks = []
        chunk_size = int(len(l)/n)
        for i in range(n+1):
            chunks.append(l[i*chunk_size:(i+1)*chunk_size])
        return chunks
    
    import random
    #impaths = sorted(glob.glob(os.path.join(settings.TEST_DIR,'original','*')))
    lo = sorted(glob.glob(os.path.join(settings.TRAIN_HEATMAP_DIR,'*')))
    impaths = []
    print(len(lo))
    for impath in lo:
        filename = utils.get_file_name_part(impath)
        nclions = int(filename.split('clions')[0])
        if nclions == 0 and random.choice([0,0,1,1,1,1,1]):
            continue
        else:
            impaths.append(impath)
    print(len(impaths))

    impaths = np.array(impaths)
    impaths = chunks(impaths, cpus)
    print(999,len(impaths))
    for a in impaths:
        print(len(a))
        print(str(a)[:500])
    
    with Pool(cpus) as pool:
        pool.starmap(generate_obms2,zip(impaths))
        
def shuffle(data, labels):
    '''
    Shuffles data keeping <image,label> pairs
    together
    '''
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    return list(pd.Series(data)[indices]), list(pd.Series(labels)[indices])
    
def paths_to_images(paths):
    '''
    Converts a list of imagepaths to a list of images
    '''
    #images = [scipy.misc.imread(path) for path in paths]
    images = [np.load(path) for path in paths]
    #TODO remove resize when reading presegmented images
    ##images = [self.random_padding(img) for img in images]
    ##images = [cv2.resize(img, dsize=(224,224)) for img in images]
    #images_ = []
    #for img in images:
        #self.augmenter.randomize()
        #images_.append(self.augmenter.augment(img)[0])
    #images = images_
    images_ = []
    for img in images:
        if random.choice([True, False]):
            img = np.flipud(img)
        if random.choice([True, False]):
            img = np.fliplr(img)     
        #images_.append(np.expand_dims(img, axis = 0))
        images_.append(img)
    
    #images = [img[np.newaxis,:,:,:] for img in images]
    #images = [img[0][np.newaxis,:,:,:] for img in images]
    #images = [img[:255,:][np.newaxis,:,:,:] for img in images]
    #images = [img[np.newaxis,:,:,:] for img in images]
    #print(images_[0].shape,999)
    result = np.vstack(images_)
    ##print(result.shape,999) 
    return result    
def generate(data, labels, batch_size):
    '''
    Generate batches of data images taking as input
    a list of image paths and labels
    '''
    
    while True:
        data, labels = shuffle(data, labels)
        batches = int(len(data)/batch_size)
        #print batches                
        for batch in range(batches):
            #print batch
            #self.augmenter.randomize()
            x_image_paths = data[batch*batch_size:(batch+1)*batch_size]
            x = paths_to_images(x_image_paths)
            y = np.array(labels[batch*batch_size:(batch+1)*batch_size])
            if len(y) != batch_size:
                break
            
            #x = preprocess_input(x)
            yield((x, y)) 
'''            
import random
import pandas as pd
from sklearn.model_selection import train_test_split
obmpaths = glob.glob(os.path.join(settings.OBMS_OUTPUT_DIR,'train','*'))
y, X = [], []
for obmpath in obmpaths:   
    filename = utils.get_file_name_part(obmpath)
    #print(im.shape,im.mean(),obm.shape,obm.mean(),filename)
    nclions = int(filename.split('clions')[0])
    if nclions == 0 and random.choice([0,0,0,1,1,1,1]):
        continue
    y.append(nclions)
    X.append(obmpath)
print((len(X),89789787))
batch_size = 512
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gt = generate(X_train, y_train, batch_size)
gv = generate(X_test, y_test, batch_size)
#xb, yb = g.__next__()
#print(xb.shape, yb.shape, xb.mean(), yb.mean())
'''


generate_obms_fast()
'''
import keras
import math
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(80,80,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

callbacks_list = []
weights_name = "obm_regressor-ep{epoch:03d}-tloss{loss:.4f}-vloss{val_loss:.4f}.hdf5"
weights_path = os.path.join(settings.WEIGHTS_DIR, weights_name)
checkpoint = keras.callbacks.ModelCheckpoint(
    weights_path,
    monitor = 'val_loss',
    verbose=1,
    save_best_only = True,
    mode = 'min')
callbacks_list.append(checkpoint)
  
#TODO get unqie_instances automatically 
unique_instances = 27000
# Train
steps_per_epoch = math.ceil(0.8*unique_instances/batch_size)
validation_steps = math.ceil(0.2*unique_instances/batch_size)

model.fit_generator(
    generator = gt,
    steps_per_epoch = steps_per_epoch, 
    epochs = 4000,
    validation_data = gv,
    validation_steps = validation_steps,
    workers = 1,
    callbacks = callbacks_list)
    
'''
    