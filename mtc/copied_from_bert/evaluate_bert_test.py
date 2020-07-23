import json
import os

import numpy as np
import pandas as pd

from mtc.settings import NLP_EXPERIMENT_PATH
from mtc.mtc_properties import KFOLDS, TASK_NAME, EPOCHS


def extract_scores(mode, folder, epoch_number):

    predictions = []

    for idx in range(KFOLDS):
        eval_file_path = os.path.join(folder, f'kfold{idx}', f'eval_results_{mode}_{epoch_number-1}.json')

        with open(eval_file_path, 'r') as file:
            eval_file = json.load(file)

        predictions.append(eval_file['pred'])

    return predictions


def evaluate_bert_test(input_folder, output_folder, epoch_number=EPOCHS):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Train
    predictions_train = []
    for pred in extract_scores('dev', input_folder, epoch_number):
        predictions_train += pred

    df = pd.DataFrame(predictions_train)
    df.to_csv(os.path.join(output_folder, 'train_scores.csv'), index=False, header=None)

    # Test
    predictions_test = extract_scores('test', input_folder, epoch_number)
    predictions_test = np.asarray(predictions_test)

    print(f'[test] mean std = {np.mean(np.std(predictions_test, axis=0))}')

    df = pd.DataFrame(np.mean(predictions_test, axis=0))
    df.to_csv(os.path.join(output_folder, 'test_scores.csv'), index=False, header=None)


if __name__ == "__main__":
    experiment_name = 'preprocessed_data_2019-08-14_15-54-19_biobert_pretrain_output_all_notes_150000_1480703'
    folder = os.path.join(NLP_EXPERIMENT_PATH, TASK_NAME, experiment_name)
    epoch_number = 11

    evaluate_bert_test(folder, '.', epoch_number)
