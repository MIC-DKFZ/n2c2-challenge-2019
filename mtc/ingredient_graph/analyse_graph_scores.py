from pathlib import Path

import pandas as pd
import numpy as np

from mtc.settings import NLP_EXPERIMENT_PATH


epsilon = 0.001


def analyse_graph_scores(time_stamp):
    submission_folder = Path(Path(NLP_EXPERIMENT_PATH) / 'submission_generation' / time_stamp)
    step2_test = pd.read_csv(submission_folder / 'normal' / 'step2_test_scores.csv', header=None).to_numpy().flatten()
    step4_test = pd.read_csv(submission_folder / 'normal' / 'step4_test_scores.csv', header=None).to_numpy().flatten()

    old_scores = step2_test[abs(step2_test - step4_test) > epsilon]
    new_scores = step4_test[abs(step2_test - step4_test) > epsilon]

    print('Effect of the score replacement in the test set')
    print(f'n = {len(old_scores)}')
    print(f'before replacement: (mean,std) = {round(np.mean(old_scores), 2), round(np.std(old_scores), 2)}')
    print(f'after replacement: (mean,std) = {round(np.mean(new_scores), 2), round(np.std(new_scores), 2)}\n')


def ratio_graph_scores(time_stamp):
    submission_folder = Path(Path(NLP_EXPERIMENT_PATH) / 'submission_generation' / time_stamp)

    print('Graph score ratio')
    for run_type in ['train', 'test']:
        step2 = pd.read_csv(submission_folder / 'normal' / f'step2_{run_type}_scores.csv', header=None).to_numpy().flatten()
        step4 = pd.read_csv(submission_folder / 'normal' / f'step4_{run_type}_scores.csv', header=None).to_numpy().flatten()

        scores = step2[abs(step2 - step4) > epsilon]
        print(f'n_{run_type} = {len(scores)} / {len(step2)} = {len(scores) / len(step2)}')


if __name__ == '__main__':
    time_stamp = '03_12_2020_20_18_37_original_data'
    analyse_graph_scores(time_stamp)
    ratio_graph_scores(time_stamp)
