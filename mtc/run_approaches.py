import os
import shutil
import subprocess
from pathlib import Path

from mtc.challenge_pipelines.example_run_step1 import example_run_step1
from mtc.challenge_pipelines.preprocess_data import generate_preprocessed_data
from mtc.copied_from_bert.evaluate_bert_test import evaluate_bert_test
from mtc.copied_from_bert.run import prepare_k_folds
from mtc.ingredient_graph.eval_graph import eval_graph
from mtc.ingredient_graph.ingredients import ingredients
from mtc.settings import NLP_EXPERIMENT_PATH
from mtc.validate_submissions import validate_submissions


def generate_bert_base(base_folder):
    base_folder.mkdir(parents=True, exist_ok=True)

    #######################
    # run.py (generates BERT scores for each fold)
    #######################
    output_folder = base_folder / '1_run'

    env = os.environ.copy()
    env['OUTPUT_FOLDER'] = output_folder
    env['USE_FEATURES'] = 'no'
    os.chdir('copied_from_bert')
    retcode = subprocess.call('python run.py', shell=True, env=env)
    assert retcode == 0, 'An error occurred while running BERT'
    os.chdir('..')


    #######################
    # evaluate_bert_test.py (combines BERT scores from kfolds, applies kfold ensembling on test set)
    #######################
    input_folder = sorted([folder for folder in output_folder.iterdir() if folder.is_dir()])[-1]  # The longest folder contains the training result
    output_folder = base_folder / '2_evaluate_bert_test'
    evaluate_bert_test(input_folder, output_folder)

    shutil.copy2(output_folder / 'train_scores.csv', base_folder / 'step1_train_scores.csv')
    shutil.copy2(output_folder / 'test_scores.csv', base_folder / 'step1_test_scores.csv')


def generate_normal(base_folder):
    base_folder.mkdir(parents=True, exist_ok=True)

    #######################
    # example_run_step1.py (generates features for BERT)
    #######################
    output_folder = base_folder / '1_example_run_step1'
    example_run_step1(output_folder)


    #######################
    # run.py (generates BERT scores for each fold)
    #######################
    input_folder = output_folder
    output_folder = base_folder / '2_run'

    env = os.environ.copy()
    env['FEATURE_PATH'] = input_folder
    env['OUTPUT_FOLDER'] = output_folder
    os.chdir('copied_from_bert')
    retcode = subprocess.call('python run.py', shell=True, env=env)
    assert retcode == 0, 'An error occurred while running BERT'
    os.chdir('..')


    #######################
    # evaluate_bert_test.py (combines BERT scores from kfolds, applies kfold ensembling on test set)
    #######################
    input_folder = sorted([folder for folder in output_folder.iterdir() if folder.is_dir()])[-1]  # The longest folder contains the training result
    output_folder = base_folder / '3_evaluate_bert_test'
    evaluate_bert_test(input_folder, output_folder)

    shutil.copy2(output_folder / 'train_scores.csv', base_folder / 'step1_train_scores.csv')
    shutil.copy2(output_folder / 'test_scores.csv', base_folder / 'step1_test_scores.csv')


    #######################
    # example_run_step2.py (voting regression)
    #######################
    input_folder = output_folder
    output_folder = base_folder / '4_example_run_step2'

    env = os.environ.copy()
    env['BERT_SCORES_PATH'] = input_folder
    env['OUTPUT_FOLDER'] = output_folder
    os.chdir('challenge_pipelines')
    retcode = subprocess.call('python example_run_step2.py', shell=True, env=env)
    assert retcode == 0, 'An error occurred while running the voting regression'
    os.chdir('..')

    shutil.copy2(output_folder / '0_dev_prediction.csv', base_folder / 'step2_train_scores.csv')
    shutil.copy2(output_folder / '0_test_prediction.csv', base_folder / 'step2_test_scores.csv')


    #######################
    # ingredients.py (extract ingredients from sentences)
    #######################
    output_folder = base_folder / '5_ingredients'

    ingredients(output_folder)

    # Generate special preprocessing and folds for the graph
    generate_preprocessed_data(['ContractionExpander', 'NumberUnifier', 'SpellingCorrector', 'MedicationRemover', 'SentenceTokenizer', 'WordTokenizer'], params=None, output_folder=output_folder, folder_name='preprocessed_data_ingredients')
    prepare_k_folds(output_folder / 'preprocessed_data_ingredients', k=10)


    #######################
    # eval_graph.py (medication graph)
    #######################
    ingredients_folder = output_folder
    input_scores_folder = base_folder / '4_example_run_step2'
    output_folder = base_folder / '6_eval_graph'
    eval_graph(ingredients_folder, input_scores_folder, output_folder)

    shutil.copy2(output_folder / 'tablet_similarity_train.csv', base_folder / 'step4_train_scores.csv')
    shutil.copy2(output_folder / 'tablet_similarity_test.csv', base_folder / 'step4_test_scores.csv')


def generate_heads(base_folder):
    base_folder.mkdir(parents=True, exist_ok=True)

    #######################
    # example_run_step1.py (generates features for BERT)
    #######################
    output_folder = base_folder / '1_example_run_step1'
    example_run_step1(output_folder)


    #######################
    # run.py (generates BERT scores for each fold)
    #######################
    input_folder = output_folder
    output_folder = base_folder / '2_run'

    env = os.environ.copy()
    env['FEATURE_PATH'] = input_folder
    env['OUTPUT_FOLDER'] = output_folder
    os.chdir('copied_from_bert')
    retcode = subprocess.call('python run_m_heads.py', shell=True, env=env)
    assert retcode == 0, 'An error occurred while running BERT'
    os.chdir('..')


    #######################
    # evaluate_bert_test.py (combines BERT scores from kfolds, applies kfold ensembling on test set)
    #######################
    input_folder = sorted([folder for folder in output_folder.iterdir() if folder.is_dir()])[-1]  # The longest folder contains the training result
    output_folder = base_folder / '3_evaluate_bert_test'
    evaluate_bert_test(input_folder, output_folder)

    train_file = output_folder / 'train_scores.csv'
    test_file = output_folder / 'test_scores.csv'

    shutil.copy2(train_file, base_folder / 'step1_train_scores.csv')
    shutil.copy2(test_file, base_folder / 'step1_test_scores.csv')

    # Rename files for the next step
    train_file.rename(train_file.with_name('0_dev_prediction.csv'))
    test_file.rename(test_file.with_name('0_test_prediction.csv'))


    #######################
    # ingredients.py (extract ingredients from sentences)
    #######################
    output_folder = base_folder / '4_ingredients'

    ingredients(output_folder)

    # Generate special preprocessing and folds for the graph
    generate_preprocessed_data(['ContractionExpander', 'NumberUnifier', 'SpellingCorrector', 'MedicationRemover', 'SentenceTokenizer', 'WordTokenizer'], params=None, output_folder=output_folder, folder_name='preprocessed_data_ingredients')
    prepare_k_folds(output_folder / 'preprocessed_data_ingredients', k=10)


    #######################
    # eval_graph.py (medication graph)
    #######################
    ingredients_folder = base_folder / '4_ingredients'
    input_scores_folder = base_folder / '3_evaluate_bert_test'
    output_folder = base_folder / '4_eval_graph'
    eval_graph(ingredients_folder, input_scores_folder, output_folder)

    shutil.copy2(output_folder / 'tablet_similarity_train.csv', base_folder / 'step4_train_scores.csv')
    shutil.copy2(output_folder / 'tablet_similarity_test.csv', base_folder / 'step4_test_scores.csv')


base_folder = Path(os.environ.get('BASE_FOLDER'))

generate_bert_base(base_folder / 'bert_base')
generate_normal(base_folder / 'normal')
generate_heads(base_folder / 'heads')

# Validate submissions
print()
validate_submissions(base_folder)
