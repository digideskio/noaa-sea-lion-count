import csv
import os
import settings

categories = ['test_id', 'adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']

def generate_submission(counts, suffix = ""):
    """
    Generate a submission file, adding a descriptive suffix (if given) to the file name.
    :param counts: a dictionary with test_id keys and a list of all the counts (in the above order)
    """
    
    if not os.path.exists(settings.SUBMISSIONS_OUTPUT_DIR):
        os.makedirs(settings.SUBMISSIONS_OUTPUT_DIR)
    
    if suffix == "":
        filename = "submission" + "-" + settings.get_timestamp() + ".csv"
    else:
        filename = "submission_" + suffix + "-" + settings.get_timestamp() + ".csv"
    
    with open(os.path.join(settings.SUBMISSIONS_OUTPUT_DIR, filename), 'w', newline='') as submfile:
        submwriter = csv.writer(submfile)
        submwriter.writerow(categories)
       
        for id in sorted(counts.keys()):
            row = [id] + [int(round(count)) for count in counts[id]]
            submwriter.writerow(row)
