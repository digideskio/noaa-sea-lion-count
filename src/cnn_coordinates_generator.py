    

import os
import time
import sys
#import network
import utils
from matplotlib import pyplot as plt
import scipy
import glob
import settings
#import data
from scipy.misc import imresize, imread
import cv2
import numpy as np
from multiprocessing import Pool
#%matplotlib inline
from itertools import chain, islice,groupby, count
import utils
import keras
import math
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

    
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
        test_image_original = scipy.misc.imread(impath)
        meta = {'filename':utils.get_file_name_part(impath)}
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
        
def generate_obms_fast():
    cpus = 20
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
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
        

def get_model():
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
    return model
#xs = np.expand_dims(np.array(list(range(300,8000))), axis = 1)
#ys = m.predict(xs)
#plt.figure()
#plt.plot(np.squeeze(ys))
#plt.show()
def get_points(crop, pred_count):
    image = crop.copy()
    npoints = pred_count
    yprobs = np.sum(image, axis=0)
    yprobs = yprobs**3
    ym = sum(yprobs)
    yprobsn = [float(i)/ym for i in yprobs]
    xprobs = np.sum(image, axis=1)
    xprobs = xprobs**3
    xm = sum(xprobs)
    xprobsn = [float(i)/xm for i in xprobs]
    
    options = list(range(window_size))
    
    proximity_threshold = 9
    
#    xcoords = np.random.choice(options,npoints,p=xprobsn)
#    ycoords = np.random.choice(options,npoints,p=yprobsn)
    points = []
    trials = 0
    max_trials = 10000
    dist = 1000
    while len(points) < npoints:
        x = np.random.choice(options,1,p=xprobsn)[0]
        y = np.random.choice(options,1,p=yprobsn)[0]
        most_close = 1000
        for p in points:
            diff = np.array([x,y]) - np.array([p[0],p[1]])
            dist = np.linalg.norm(diff)
            #print(p,'and',x,y,'are',dist)
            if dist < most_close:
                most_close = dist
            if most_close < proximity_threshold:
                break
        if dist > proximity_threshold:
            points.append((x,y))
        if trials > max_trials:
            break
        trials +=1
    
#    for x,y in points:
#        cv2.circle(image,(y,x),5,(1,1,1),0)
#    image += crop
#    plt.figure()
#    plt.imshow(image, cmap = 'gray')
#    plt.show()
    return points

model = get_model()
#weights_name = 'obm_regressor-ep001-tloss6.4490-vloss6.3203.hdf5'
weights_name = 'obm_regressor-ep016-tloss13.2885-vloss12.0621.hdf5'
weights_filepath = os.path.join(settings.WEIGHTS_DIR,weights_name)
model.load_weights(weights_filepath)

obm_paths = sorted(glob.glob(os.path.join(settings.OBMS_OUTPUT_DIR,'*')))
threshold = 0.3
window_size = 80
import pandas as pd
import numpy as np
np.random.shuffle(obm_paths)
for obm_path in obm_paths:
    #obm_path = '/vol/tensusers/vgarciacazorla/MLP/noaa-sea-lion-count/output/obms/16435_obm.h5.npy'
    fname= utils.get_file_name_part(obm_path)
    if os.path.isfile('image_samples/coords_images/'+fname.split('_')[0]+'_test.jpg'):
        continue
    else:
        pass#print("MISSING ",fname)
    if os.path.isfile(os.path.join(settings.CNN_COORDINATES_DIR,fname.split('_')[0]+'_coords.csv')):
        #print("Skippppp",fname)
        pass
    print(1)
    test_image = scipy.misc.imread(os.path.join(settings.TRAIN_DIR,'original',fname.split('_')[0]+'.jpg'))
    print(2)
    
    
    aux_height = test_image.shape[0] - test_image.shape[0] % 400 + 400
    aux_width = test_image.shape[1] - test_image.shape[1] % 400 + 400
    
    ptest_image = np.zeros((aux_height,aux_width,3))
    ptest_image[:test_image.shape[0], :test_image.shape[1]] = test_image
    
    obm = np.load(obm_path)
    #obm = obm / threshold
#    plt.figure()
#    plt.imshow(obm, cmap = 'gray')
#    plt.title(str(obm.max())+str(obm.shape))
#    plt.show()
    
    aux_height = obm.shape[0] - obm.shape[0] % window_size + window_size
    aux_width = obm.shape[1] - obm.shape[1] % window_size + window_size
    
    padded = np.zeros((aux_height,aux_width))
    padded[:obm.shape[0], :obm.shape[1]] = obm
    #plt.figure()
    #plt.imshow(padded, cmap = 'gray')
    #plt.title(str(padded.max())+str(padded.shape))
    #plt.show()
    ratio = 400/80
    nrows = int(padded.shape[0] / window_size)
    ncolumns = int(padded.shape[1] / window_size)
    
    df = []
    print(4)
    for row in range(nrows):
        for column in range(ncolumns):
            xcoord = column*window_size
            ycoord = row*window_size
            rxcoord = int(xcoord * ratio)
            rycoord = int(ycoord * ratio)
            crop = utils.crop_image(padded,(xcoord, ycoord), window_size)
            suma = crop.sum()
            #print(suma)
            if suma < 0.7 or 0:
                continue
            else:
                aux = np.expand_dims(crop, axis = 0)
                aux = np.expand_dims(aux, axis = 3)
#                plt.figure()
#                plt.imshow(crop, cmap='gray')
#                plt.title(str((ycoord, xcoord, suma)))
#                plt.show()
                pred_count = model.predict(aux)[0][0]
                original_crop = utils.crop_image(ptest_image,(rxcoord, rycoord), 400)
                if pred_count < 1 or 0:
                    continue
                points = get_points(crop, round(pred_count))
                for x,y in points:
                    
                    r = {'filename':fname.split('_')[0]+'.jpg',
                           'y_coord':float(rycoord + y*ratio),
                           'x_coord':float(rxcoord + x*ratio)}
                    #print(r)
                    df.append(r)
#                image1 = crop.copy()
#                imager = original_crop.copy()
#                for x,y in points:
#                    cv2.circle(image1,(x,y),5,(1,1,1),0)
#                    cv2.circle(imager,(int(x*ratio),int(y*ratio)),50,(0,0,0),-1)
#                image1 += crop
#                imager += original_crop
#                #assert imager.shape == (400, 400, 3)
#                plt.figure()
#                plt.imshow(image1, cmap = 'gray')
#                plt.title(str((xcoord, ycoord,suma, pred_count)))
#                plt.show()
##
#                plt.figure()
#                plt.imshow(imager, cmap = 'gray')
#                plt.title(str((rxcoord, rycoord,suma, pred_count)))
#                plt.show()
#                print("_________________")
                
    df = pd.DataFrame(df)
    dfpath = os.path.join(settings.CNN_COORDINATES_DIR,fname.split('_')[0]+'_coords.csv')
    df.to_csv(dfpath)
    


    image = ptest_image.copy()
    for ix, row in df.iterrows():
        cv2.circle(image,(int(row['x_coord']),int(row['y_coord'])),20,(200,0,0),-1)
    scipy.misc.imsave('image_samples/coords_images/'+fname.split('_')[0]+'_test.jpg', image)
    '''
    plt.figure()
    plt.imshow(image, cmap = 'gray')
    plt.title(str(df.shape))
    plt.show()
    '''
    print("______________________")
    
    #print(df)
