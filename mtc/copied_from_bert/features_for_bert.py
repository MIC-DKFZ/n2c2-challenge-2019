import os
from typing import List
import re

import numpy as np
from mtc.core.experiment import Measuring
from mtc.settings import NLP_EXPERIMENT_PATH, NLP_RAW_DATA
from mtc.helpers.file_management import len_sts_data

input_folder = os.environ.get('FEATURE_PATH', os.path.join(NLP_EXPERIMENT_PATH, 'pickles_for_bert'))
use_features = os.environ.get('USE_FEATURES', 'yes')


def load_features(pickle_folder, sts_data_path):
    measuring = Measuring([], pickle_folder)
    measuring.set_sts_data_dict(sts_data_path)
    measuring.load_feature_matrix()
    return measuring


def get_features_for_bert_size():
    if use_features == 'yes':
        pickle_folder = os.path.join(input_folder, 'train')
        measuring = load_features(pickle_folder, os.path.join(NLP_RAW_DATA, 'n2c2', 'clinicalSTS2019.train.txt'))

        X, y, raw_sentences_a, raw_sentences_b = measuring()

        return X.shape[1]
    else:
        return 0


def add_features_to_bert(examples) -> List:
    if use_features == 'yes':
        # Applying similarity measures and saving the sentences object

        guids = np.array([example.guid for example in examples])

        match = re.search(r'(\w+)-', guids[0])
        assert match, f'Could not read guid {guids[0]}'
        mode = match.group(1)

        n_train = len_sts_data('clinicalSTS2019.train.txt')

        id_list = []
        id_shift = 0
        if mode == 'train' or mode == 'dev':
            train_or_test_folder = 'train'
            sts_data_path = os.path.join(NLP_RAW_DATA, 'n2c2', 'clinicalSTS2019.train.txt')
        elif mode == 'test':
            train_or_test_folder = 'test'
            sts_data_path = os.path.join(NLP_RAW_DATA, 'n2c2', 'clinicalSTS2019.test.txt')

            # The ids in bert continue after the training set but here we need the indices to use as array index
            id_shift = n_train
        else:
            assert False

        for guid in guids:
            match = re.search(r'-(\d+)', guid)
            assert match, f'Could not extract id from {guid}'

            id = int(match.group(1)) - id_shift
            id_list.append(id)

        pickle_folder = os.path.join(input_folder, train_or_test_folder)
        measuring = load_features(pickle_folder, sts_data_path)

        X, y, raw_sentences_a, raw_sentences_b = measuring()

        examples_labels = np.array([float(example.label) for example in examples])
        assert all(examples_labels == y[id_list]), 'The labels between bert and the features do not match up'

        return X[id_list]
    else:
        return [[] for _ in range(len(examples))]


if __name__ == '__main__':
    get_features_for_bert_size()
