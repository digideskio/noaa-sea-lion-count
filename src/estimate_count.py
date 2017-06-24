import csv
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from collections import OrderedDict
import os
from os import listdir
from os.path import isfile, join

'''
Please make sure to use computeCount() to compute the counts.
It is located at the bottom of the file.
'''

def loadPriors(dataCSVs):
    cluster_data = {}
    outlier_data = {}
    count_data = {}
    count_group_data = {}
    cluster_group_data = {}
    for csvFile in dataCSVs:
        for row in csv.reader(open(csvFile),delimiter=","):
            total = row[5]
            if total != "total":
                if csvFile.endswith("cluster_priors.csv"):
                    if total not in cluster_data.keys():
                        cluster_data[total] = row
                elif csvFile.endswith("outlier_priors.csv"):
                    if total not in outlier_data.keys():
                        outlier_data[total] = row
                elif csvFile.endswith("sealion_count_priors.csv"):
                    if total not in count_data.keys():
                        count_data[total] = row
                elif csvFile.endswith("sealion_group_count_priors.csv"):
                    if total not in count_group_data.keys():
                        count_group_data[eval(total)] = row
                elif csvFile.endswith("cluster_group_priors.csv"):
                    if total not in cluster_group_data.keys():
                        cluster_group_data[eval(total)] = row
    return (cluster_data, outlier_data, count_data, count_group_data, cluster_group_data)


def compare(args):
    coord, coord2 = args
    dst = distance.euclidean(coord,coord2)
    return dst
    
def computeDistances(coords):
    print("compute distances")
    distances = np.zeros((coords.shape[0],coords.shape[0]))
    for i, coord in enumerate(coords):
        all_coords = [(coord,coord2) for coord2 in coords]
        dists = list(map(compare, all_coords))
        distances[i, :] = dists
    return distances

def readLocations(locations):
    csv_data = {}
    csv_columns = defaultdict(list)
    filenames = set()
    with open(locations) as f:
        reader = csv.DictReader(f)
        data = list(reader)
        for row in data:
            filename = row['filename']
            filenames.add(filename)
        for filename in filenames:
            for row in data:
                if row['filename'] == filename:
                    for (k,v) in row.items():
                        if k == 'y_coord' or k == 'x_coord':
                            csv_columns[k].append(float(v))
                        else:
                            csv_columns[k].append(v)
            csv_data[filename] = csv_columns
            csv_columns = defaultdict(list)

    return csv_data

def addEstimates(counts, cluster_size, nearest_size, priors,pups):
    if pups == True:
        counts['pups'] = round(counts['pups'] + (float(priors[nearest_size]['pups']) * cluster_size))
    else:
        counts['pups'] = round(counts['pups'] + ((cluster_size/(1-float(priors[nearest_size]['pups'])))-cluster_size))
    counts['juveniles'] = round(counts['juveniles'] + (float(priors[nearest_size]['juveniles']) * cluster_size))
    counts['adult_females'] = round(counts['adult_females'] + (float(priors[nearest_size]['adult_females']) * cluster_size))
    counts['subadult_males'] = round(counts['subadult_males'] + (float(priors[nearest_size]['subadult_males']) * cluster_size))
    counts['adult_males'] = round(counts['adult_males'] + (float(priors[nearest_size]['adult_males']) * cluster_size))
    return counts

def addOutlierEstimates(counts, cluster_size, priors,pups):
    if pups == True:
        counts['pups'] = round(counts['pups'] + (float(priors['pups']) * cluster_size))
    else:
        counts['pups'] = round(counts['pups'] + ((cluster_size/(1-float(priors['pups'])))-cluster_size))
    counts['juveniles'] = round(counts['juveniles'] + (float(priors['juveniles']) * cluster_size))
    counts['adult_females'] = round(counts['adult_females'] + (float(priors['adult_females']) * cluster_size))
    counts['subadult_males'] = round(counts['subadult_males'] + (float(priors['subadult_males']) * cluster_size))
    counts['adult_males'] = round(counts['adult_males'] + (float(priors['adult_males']) * cluster_size))
    return counts

def initiateResults():
    results = {}
    results['pups'] = 0
    results['juveniles'] = 0
    results['adult_females'] = 0
    results['subadult_males'] = 0
    results['adult_males'] = 0
    return results

def determineGroup(cluster_size,groups):
    groupNumber = (0,0)
    for i, group in enumerate(groups):
        (min,max) = group
        if min <= cluster_size <= max:
            groupNumber = group
        elif (cluster_size > max) and (max == 1400):
            groupNumber = group
            
    return groupNumber

def estimateCount(clusters,cluster_data,outlier_data,groupBool,pups):
    clusters = clusters
    clusterPriors = cluster_data
    outlierPriors = outlier_data
    results = initiateResults()
    
    unique_labels = set(clusters)
    for k in unique_labels:
        if k != -1:
            cluster_size = (clusters == k).sum()
            cluster_sizes = cluster_data.keys()
            if groupBool == True:
                nearest_size = determineGroup(cluster_size,cluster_sizes)
            else:
                nearest_size = str(float(min(cluster_sizes, key=lambda x:abs(float(x)-cluster_size))))
            results = addEstimates(results, cluster_size,nearest_size,clusterPriors, pups)
      
    for k in unique_labels:
        if k == -1:
            cluster_size = (clusters == k).sum()
            results = addOutlierEstimates(results, cluster_size, outlierPriors, pups)
        
    return results
    
def estimateImageCount(sealionCount,countPriors,groupBool,pups):
    herdSizes = countPriors.keys()
    if groupBool == True:
        nearestSize = determineGroup(sealionCount,herdSizes)
    else:
        nearestSize = str(float(min(herdSizes, key=lambda x:abs(float(x)-sealionCount))))
    results = initiateResults()

    results = addEstimates(results,sealionCount,nearestSize,countPriors,pups)

    return results

def makeStats(row):
    row_info = {}
    pupsPrior = row[0]
    juvenilesPrior = row[1]
    adult_femalesPrior = row[2]
    subadult_malesPrior = row[3]
    adult_malesPrior = row[4]
    row_info['pups'] = pupsPrior
    row_info['juveniles']= juvenilesPrior
    row_info['adult_females'] = adult_femalesPrior
    row_info['subadult_males'] = subadult_malesPrior
    row_info['adult_males'] = adult_malesPrior
    return row_info

def loadCount(data):
    priors = {}
    for size, row in data.items():
        row_info = makeStats(row)
        priors[size] = row_info

    return priors

def loadOutlierCount(outlier_data):
    priors = {}
    for cluster_size, row in outlier_data.items():
        priors['pups'] = row[0]
        priors['juveniles'] = row[0]
        priors['adult_females'] = row[0]
        priors['subadult_males'] = row[0]
        priors['adult_males'] = row[0]

    return priors

def clusterSeaLions(locations,cluster_data,outlier_data,groupBool,pups):
    clusterPriors = loadCount(cluster_data)
    outlierPriors = loadOutlierCount(outlier_data)
    csv_data = readLocations(locations)
    results = {}
    for filename in csv_data.keys():
        y_coords = np.array(csv_data[filename]['y_coord'])
        x_coords = np.array(csv_data[filename]['x_coord'])
        coords = np.column_stack((y_coords,x_coords))
        distances = computeDistances(coords)
        cls = DBSCAN(metric='precomputed', min_samples=3, eps=200)
        y = cls.fit_predict(distances)
        
        num_clusters = len(set(y))
        print('Number of clusters: {}'.format(num_clusters))

        counts = estimateCount(y,clusterPriors,outlierPriors,groupBool,pups)
        results[filename] = counts

    return results

def countSeaLions(locations,count_data,groupBool,pups):
    countPriors = loadCount(count_data)
    csv_data = readLocations(locations)
    results = {}
    for filename in csv_data.keys():
        sealionCount = len(csv_data[filename]['y_coord'])
        counts = estimateImageCount(sealionCount,countPriors,groupBool,pups)   
        results[filename] = counts
    return results

def obtainFiles():
    filedir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/train/priors'))
    files = [join(filedir,f) for f in listdir(filedir) if isfile(join(filedir,f))]
    return files

def prepareResultsData(results):
    data = []
    labels = ['test_id', 'pups', 'juveniles', 'adult_females', 'subadult_males', 'adult_males']
    data.append(labels)
    filenames = results.keys()
    for filename in filenames:
        counts = results[filename]
        pups = str(int(counts['pups']))
        juveniles = str(int(counts['juveniles']))
        adult_females = str(int(counts['adult_females']))
        subadult_males = str(int(counts['subadult_males']))
        adult_males = str(int(counts['adult_males']))
        entry = [filename, pups, juveniles, adult_females, subadult_males, adult_males]
        data.append(entry)

    return data

def write2csv(name, data):
    ofile = open(name,'w', newline='')
    writer = csv.writer(ofile, delimiter=',', quotechar='"')
    writer.writerows(data)
    ofile.close()

'''
This is the function that should be used to compute count estimates.
:params: locations: a .csv file exactly like coordinates.csv but without the category column.
:params: groupBool: a boolean to decide if group priors should be used or not.
:params: cluster: a boolean to decide if the cluster functionality should be used or just a global count of the sealions.
:params: pups: a boolean to decide if pups were considered in the count/locations or not.

:return: results: a dictionary with as keys the filenames and as values a dictionary. This dictionary has as keys the different categories and as values the counts.
'''
def computeCount(locations,groupBool,cluster,pups):
    #dataCSVs = ['cluster_priors.csv', 'outlier_priors.csv', 'sealion_count_priors.csv', 'sealion_group_count_priors.csv', 'cluster_group_priors.csv' ] 
    dataCSVs = obtainFiles()
    (cluster_data, outlier_data, count_data, count_group_data, cluster_group_data) = loadPriors(dataCSVs)
    if groupBool == True:
        if cluster == True:
            results = clusterSeaLions(locations,cluster_group_data,outlier_data,groupBool,pups)
        else:
            results = countSeaLions(locations,count_group_data,groupBool,pups)
    else:
        if cluster == True:
            results = clusterSeaLions(locations,cluster_data,outlier_data,groupBool,pups)
        else:
            results = countSeaLions(locations,count_data,groupBool,pups)
    
    results_data = prepareResultsData(results)
    write2csv("counts_groupBool=%s_cluster=%s_pups=%s.csv"%(str(groupBool),str(cluster),str(pups)),results_data)



    



