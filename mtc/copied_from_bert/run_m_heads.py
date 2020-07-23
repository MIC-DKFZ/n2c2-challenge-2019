import json
import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

from mtc.challenge_pipelines.preprocess_data import generate_preprocessed_data
from mtc.helpers.MeasureTime import MeasureTime
from mtc.settings import NLP_RAW_DATA, NLP_EXPERIMENT_PATH, NLP_MODELS_PATH
from mtc.mtc_properties import KFOLDS, TASK_NAME, EPOCHS
from mtc.helpers.JSONNumpyEncoder import JSONNumpyEncoder

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def prepare_k_folds(input_dir, k):
    """
    Generates kfold data based on a preprocessed data file. If the folders already exist, the the files are regenerated.

    :param input_dir: Path to the folder with the preprocessed data.
    """
    df_data_train = pd.read_csv(os.path.join(input_dir, "preprocessed_data_train.tsv"), delimiter='\t')
    df_data_test = pd.read_csv(os.path.join(input_dir, "preprocessed_data_test.tsv"), delimiter='\t')

    kf = KFold(n_splits=k)
    for idx, (train, dev) in enumerate(kf.split(df_data_train)):
        outdir = os.path.join(input_dir, f'kfold%s' % idx)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        df_data_train.loc[df_data_train['index'].isin(train)].to_csv(os.path.join(outdir, 'train.tsv'), sep='\t', index=False)
        df_data_train.loc[df_data_train['index'].isin(dev)].to_csv(os.path.join(outdir, 'dev.tsv'), sep='\t', index=False)
        df_data_test.to_csv(os.path.join(outdir, 'test.tsv'), sep='\t', index=False)


def run_classifier(input_dir, output_dir, model_type, model, remove_model=True):
    # Per default, the configuration has the same name as the model
    config = model

    # If a model directory is given, search for a json file
    if os.path.isdir(model):
        files = os.listdir(model)
        for file in files:
            if file.endswith('.json'):
                config = os.path.join(model, file)

    for idx in range(KFOLDS):
        command = '''
        python run_glue_m_heads.py \
              --model_type {model_type} \
              --task_name {task_name} \
              --do_train \
              --do_eval \
              --do_lower_case \
              --data_dir {input_dir}/kfold{idx} \
              --model_name_or_path {model} \
              --config_name {config} \
              --max_seq_length 128 \
              --m_heads 4 \
              --learning_rate 2e-5 \
              --num_train_epochs {epochs} \
              --output_dir {output_dir}/kfold{idx} \
              --logging_steps 0 \
              --save_steps 0
        '''.format(idx=idx, model_type=model_type, task_name=TASK_NAME, output_dir=output_dir, input_dir=input_dir, model=model, config=config, epochs=EPOCHS)

        # Store the calling configuration at least once
        if idx == 0:
            with open(os.path.join(output_dir, 'arguments.txt'), 'w', encoding='utf-8') as file:
                file.write(command)

        subprocess.call(command, shell=True)

        if remove_model:
            # Clean up: remove the trained model created for each fold since it can be quite large
            os.remove(os.path.join(output_dir, 'kfold' + str(idx), 'pytorch_model.bin'))


def eval_k_folds(input_dir, output_dir, experiment_name):
    df_data = pd.read_csv(os.path.join(input_dir, "preprocessed_data_train.tsv"), delimiter='\t')
    df_data = df_data.set_index('index')

    for idx in range(KFOLDS):
        filename = os.path.join(output_dir, 'kfold%s' % idx, 'eval_results_dev.json')
        if not os.path.exists(filename):
            print('WARNING: The folder {output_dir} contains only {k_folder} folders but the variable KFOLDS specifies {k_global} folds. Make sure that this is what you want.'.format(output_dir=output_dir, k_folder=idx-1, k_global=KFOLDS))
            break

        with open(filename, 'r') as file:
            eval_data = json.load(file)

        indices = eval_data['index']
        test_labels = eval_data['test_labels']
        predictions = eval_data['pred']

        for index, pred, label in zip(indices, predictions, test_labels):
            df_data.at[index, 'pred'] = pred
            df_data.at[index, 'test_labels'] = label

    # The Pearson correlation coefficient must be calculated based on all test scores
    pearson = pearsonr(df_data['pred'], df_data['test_labels'])[0]
    print('pearson = {}'.format(pearson))

    df_data['diff'] = df_data['pred'] - df_data['score']
    df_data = df_data.sort_values(by=['diff'], ascending=False)
    df_data.to_csv(os.path.join(output_dir, "diff_sentences.tsv"), sep='\t')

    evaluation_dict = {
        'name': [experiment_name],
        'pearson': [pearson],
    }
    df_evaluation = pd.DataFrame(evaluation_dict)
    filename = os.path.join(output_dir, '..', 'pearson_output.csv')
    if not os.path.isfile(filename):
        df_evaluation.to_csv(filename, header=list(df_evaluation.columns), index=False, sep='\t')
    else:
        df_evaluation.to_csv(filename, mode='a', header=False, index=False, sep='\t')


def eval_steps(output_dir):
    pearsons = []
    train_losses = []
    for epoch in range(EPOCHS):
        test_labels = []
        predictions = []
        train_losses_folds = []
        for idx in range(KFOLDS):
            filename = os.path.join(output_dir, 'kfold%s' % idx, f'eval_results_dev_{epoch}.json')
            assert os.path.exists(filename)

            with open(filename, 'r') as file:
                eval_data = json.load(file)

            test_labels += eval_data['test_labels']
            predictions += eval_data['pred']
            train_losses_folds.append(eval_data['train_loss'])

        train_losses.append(np.mean(train_losses_folds))

        # The Pearson correlation coefficient must be calculated based on all test scores
        pearson = pearsonr(predictions, test_labels)[0]
        pearsons.append(pearson)

    with open(os.path.join(output_dir, 'evaluation.json'), 'w') as file:
        json.dump({
            'epochs': EPOCHS,
            'kfolds': KFOLDS,
            'pearsons': pearsons,
            'train_losses': train_losses
        }, file, indent='\t', cls=JSONNumpyEncoder)


if __name__ == "__main__":
    with MeasureTime():
        output_folder = os.environ.get('OUTPUT_FOLDER', os.path.join(NLP_RAW_DATA, TASK_NAME))

        preprocessing_name = generate_preprocessed_data(['ContractionExpander', 'NumberUnifier', 'SpellingCorrector', 'LowerCaseTransformer'], output_folder=output_folder)
        # 'ContractionExpander', 'NumberUnifier', 'SpellingCorrector', 'SentenceTokenizer', 'WordTokenizer', 'PunctuationRemover', 'LowerCaseTransformer'
        #preprocessing_name = 'preprocessed_data_2019-07-30_18-00-28'

        model_type = 'bert'
        model_name = 'biobert_pretrain_output_all_notes_150000'
        experiment_name = preprocessing_name + '_' + model_name

        #model = model_name
        model = os.path.join(NLP_MODELS_PATH, 'pretrained', 'word_embeddings', 'bert_models', model_name)

        input_dir = os.path.join(output_folder, preprocessing_name)
        output_dir = os.path.join(output_folder, experiment_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prepare_k_folds(input_dir, k=KFOLDS)
        run_classifier(input_dir, output_dir, model_type, model)
        eval_k_folds(input_dir, output_dir, experiment_name)
        eval_steps(output_dir)
