from pathlib import Path

import pandas as pd
import numpy as np

from mtc.settings import NLP_EXPERIMENT_PATH


def analyse_graph_scores(time_stamp):
    submission_folder = Path(Path(NLP_EXPERIMENT_PATH) / 'submission_generation' / time_stamp)
    step2_test = pd.read_csv(submission_folder / 'normal' / 'step2_test_scores.csv', header=None).to_numpy().flatten()
    step4_test = pd.read_csv(submission_folder / 'normal' / 'step4_test_scores.csv', header=None).to_numpy().flatten()

    old_scores = step2_test[step2_test != step4_test]
    new_scores = step4_test[step2_test != step4_test]

    print(f'n = {len(old_scores)}')
    print(f'mean scores before replacement: {round(np.mean(old_scores), 2)}')
    print(f'mean scores after replacement: {round(np.mean(new_scores), 2)}')


def ratio_graph_scores(time_stamp):
    submission_folder = Path(Path(NLP_EXPERIMENT_PATH) / 'submission_generation' / time_stamp)

    for type in ['train', 'test']:
        step2 = pd.read_csv(submission_folder / 'normal' / f'step2_{type}_scores.csv', header=None).to_numpy().flatten()
        step4 = pd.read_csv(submission_folder / 'normal' / f'step4_{type}_scores.csv', header=None).to_numpy().flatten()

        scores = step2[step2 != step4]
        print(f'n_{type} = {len(scores)} / {len(step2)} = {len(scores) / len(step2)}')

if __name__ == '__main__':
    time_stamp = '03_12_2020_20_18_37'
    analyse_graph_scores(time_stamp)
    ratio_graph_scores(time_stamp)
