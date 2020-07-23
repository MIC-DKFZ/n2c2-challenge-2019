import re
from pathlib import Path

import pandas as pd

from mtc.settings import NLP_EXPERIMENT_PATH, NLP_RAW_DATA
from scipy.stats import pearsonr


def validate_type(submission_folder, train_scores_true, test_scores_true):
    files_scores = list(submission_folder.glob('*.csv'))
    steps = [int(re.search(r'step(\d)', file.name).group(1)) for file in files_scores]
    steps = sorted(list(set(steps)))

    step_names  = {
        1: 'Enhanced BERT',
        2: 'Voting Regression',
        4: 'Medication Graph'
    }

    for step in steps:
        train_scores_pred = pd.read_csv(submission_folder / f'step{step}_train_scores.csv', header=None)[0].to_numpy()
        test_scores_pred = pd.read_csv(submission_folder / f'step{step}_test_scores.csv', header=None)[0].to_numpy()

        print(f'Train scores step {step} ({step_names[step]}): {round(pearsonr(train_scores_pred, train_scores_true)[0], 3)}')
        print(f'Test scores step {step} ({step_names[step]}): {round(pearsonr(test_scores_pred, test_scores_true)[0], 3)}')


def validate_submissions(base_folder):
    # True score labels
    data_folder = Path(NLP_RAW_DATA, 'n2c2')
    train_scores_true = pd.read_csv(data_folder / 'clinicalSTS2019.train.txt', delimiter='\t', header=None)[2].to_numpy()
    test_scores_true = pd.read_csv(data_folder / 'clinicalSTS2019.test.gs.sim.txt', header=None)[0].to_numpy()

    # Predictions
    for entry in base_folder.glob('*'):
        if entry.is_dir() and entry.name != 'input_data':
            print(f'{entry.name}:')
            validate_type(entry, train_scores_true, test_scores_true)


if __name__ == '__main__':
    base_folder = Path(NLP_EXPERIMENT_PATH) / 'submission_generation/04_28_2020_10_09_57'
    NLP_RAW_DATA = base_folder / 'input_data'
    validate_submissions(base_folder)
