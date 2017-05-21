from collections import defaultdict
import csv
import scipy.stats
import settings
import statistics

stat_functions = {'average': statistics.mean, 'geometric': scipy.stats.gmean, 'harmonic': scipy.stats.hmean, 'median': statistics.median}

def prior_counts(statistic):
    """
    Computes a given statistic of the counts in total and per category.
    :param statistic: see stat_functions above for the allowed values
    """
    
    if statistic not in stat_functions:
        raise KeyError('Invalid statistic requested: ' + statistic)
    
    with open(settings.TRAIN_COUNTS_CSV, newline='') as countsfile:
        countsreader = csv.reader(countsfile)
        header = next(countsreader)
        accumulators = [[] for _ in header[1:]]
        
        for row in countsreader:
            for item, acc in zip(row[1:], accumulators):
                acc.append(float(item))
    
    total = [item for acc in accumulators for item in acc]
    
    results = {}
    for col, acc in zip(header[1:], accumulators):
        results[col] = stat_functions[statistic](acc)
    results['total'] = stat_functions[statistic](total)
    
    return results
