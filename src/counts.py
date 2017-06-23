"""
Module for working with sea lion counts.
"""

import settings
import data
import stats
import csv

logger = settings.logger.getChild('counts')

def from_coordinates():
    d = data.Loader()
    avg_counts = stats.prior_counts('average')

    total = (
        avg_counts['adult_males']
        + avg_counts['subadult_males']
        + avg_counts['adult_females']
        + avg_counts['juveniles']
        + avg_counts['pups'])

    prior = {
        'adult_males':      avg_counts['adult_males'] / total,
        'subadult_males':   avg_counts['subadult_males'] / total,
        'adult_females':    avg_counts['adult_females'] / total,
        'juveniles':        avg_counts['juveniles'] / total,
        'pups':             avg_counts['pups'] / total,
    }
    
    logger.info("Loading test coordinates (this may take a long time)")
    coords = d.load_test_coordinates()
    logger.info("Loading test image information")
    images = d.load_original_images('test_st1')

    logger.info("Starting output writing")
    with open(settings.TEST_COUNTS_CSV, 'w', newline="\n", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=['test_id', 'adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups'])
        writer.writeheader()
        for img in images:
            id = img['m']['filename']
            n = len(coords[id]) if id in coords else 0

            counts = {
                'test_id':          id,
                'adult_males':      round(n * prior['adult_males']),
                'subadult_males':   round(n * prior['subadult_males']),
                'adult_females':    round(n * prior['adult_females']),
                'juveniles':        round(n * prior['juveniles']),
                'pups':             round(n * prior['pups']),
            }
            writer.writerow(counts)

    # Average of 74 sea lions per image in the predicted test coordinates (excluding pups)
    # Average of 85 sea lions per image in the ground truth train coordinates (including pups)


    
