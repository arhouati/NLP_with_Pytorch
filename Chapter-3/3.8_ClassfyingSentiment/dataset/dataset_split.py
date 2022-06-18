import collections
import re

import pandas as pd
import numpy as np
import argparse


# set up arguments
parser = argparse.ArgumentParser(description='Split DataSet Arguments.')
parser.add_argument('--seed', dest='seed', type=int, help='seed argument')
parser.add_argument('--train', dest='train_propostion', type=float, help='seed argument', default="0.7")
parser.add_argument('--val', dest='val_propostion', type=float, help='seed argument', default="0.2")
parser.add_argument('--test', dest='test_propostion', type=float, help='seed argument', default="0.1")

args = parser.parse_args()

# Read CSV File of Yelp Dataset
csv_file = "../data/raw_train.csv"
review_subset = pd.read_csv(csv_file, delimiter=',')

by_rating = collections.defaultdict(list)

for _, row in review_subset.iterrows():
    by_rating[row.rating].append(row.to_dict())
    # important get only a few example line for a debug purpose
    #if _ == 100:
    #    break

final_list = []
np.random.seed(args.seed)

# Split dataset to train, valid and test ensembles
for _, item_list in sorted(by_rating.items()):
    np.random.shuffle(item_list)

    # define number of lines for each subsets
    n_total = len(item_list)
    n_train = int(args.train_propostion * n_total)
    n_val = int(args.val_propostion * n_total)
    n_test = int(args.test_propostion * n_total)

    # Add split attribute to dataset's line
    for item in item_list[:n_train]:
        item['split'] = 'train'

    for item in item_list[n_train:n_train+n_val]:
        item['split'] = 'val'

    for item in item_list[n_train+n_val:n_train+n_val+n_test]:
        item['split'] = 'test'

    # add line to final list
    final_list.extend(item_list)

final_reviews = pd.DataFrame(final_list)

# Cleaning the data function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]", r" ", text)
    return text

# cleaning the data
final_reviews.text = final_reviews.text.apply(preprocess_text)

# export data to a new csv file
final_reviews.to_csv (r'../data/raw_split.csv', index = True, header=True)


