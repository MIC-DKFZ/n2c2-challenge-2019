import datetime
import json
import os
import re
import subprocess

import gensim
import pandas as pd
import numpy as np

from mtc.core.experiment import Measuring
from mtc.core.preprocessor import Preprocessor
from mtc.helpers.file_management import load_sts_data
from mtc.helpers.text_processing import TextProcessing
from mtc.settings import NLP_MODELS_PATH, NLP_RAW_DATA


def generate_preprocessing_file(folder, mode, processing_steps, params=None):
    steps = []
    for step in processing_steps:
        if step == 'ContractionExpander':
            # Check whether the correct Java version is used
            ret = subprocess.run(['java', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert bool(re.search(r'"1\.8', ret.stderr.decode('utf-8'))), 'This script requires Java 8 to be installed'

            model_path = os.path.join(NLP_MODELS_PATH, 'pretrained', 'word_embeddings', 'pubmed2018_w2v_400D',
                                      'pubmed2018_w2v_400D.bin')
            model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
            steps.append(TextProcessing(step, model))
        elif step == 'RemoveHighNGrams':
            steps.append(TextProcessing(step, params['ngrams_num'], params['n_min']))
        else:
            steps.append(TextProcessing(step))

    preprocessor = Preprocessor('SelectivePreprocessor', steps)
    sts_data = load_sts_data(os.path.join('n2c2', f'clinicalSTS2019.{mode}.txt'))

    # Preprocess data
    sentences_combined = Measuring.preprocess_sentences(preprocessor, sts_data['raw_sentences_a'] + sts_data['raw_sentences_b'])
    sent_a = sentences_combined[:len(sentences_combined) // 2]
    sent_b = sentences_combined[len(sentences_combined) // 2:]
    sent_a = [s.to_plain_string() for s in sent_a]
    sent_b = [s.to_plain_string() for s in sent_b]

    # Write data
    filename = 'preprocessed_data'
    if 'similarity_score' in sts_data:
        df = pd.DataFrame(list(zip(sent_a, sent_b, sts_data['similarity_score'])), columns=['sentence a', 'sentence b', 'score'])
    else:
        df = pd.DataFrame(list(zip(sent_a, sent_b)), columns=['sentence a', 'sentence b'])

        # The indices of the test set should start
        sts_data_train = load_sts_data(os.path.join('n2c2', 'clinicalSTS2019.train.txt'))
        len_train = len(sts_data_train['raw_sentences_a'])
        df.index = np.arange(len_train, len_train + len(sent_a))
    df.to_csv(os.path.join(folder, filename + f'_{mode}.tsv'), sep='\t', index_label='index')

    # Write configuration
    with open(os.path.join(folder, filename + f'_{mode}.json'), 'w') as file:
        json.dump(preprocessor.get_params(), file, indent='\t')

    print('Stored preprocessed data to ' + folder)


def generate_preprocessed_data(processing_steps=(
'ContractionExpander', 'NumberUnifier', 'SpellingCorrector', 'SentenceTokenizer', 'WordTokenizer', 'PunctuationRemover',
'StopWordsRemover', 'LowerCaseTransformer', 'Lemmatizer'), params=None, output_folder=None, folder_name=None):
    if folder_name is None:
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        folder_name = 'preprocessed_data_' + now

    if output_folder is None:
        output_folder = os.path.join(NLP_RAW_DATA, 'n2c2', folder_name)
        os.makedirs(output_folder)

    output_folder = os.path.join(output_folder, folder_name)
    os.makedirs(output_folder)

    generate_preprocessing_file(output_folder, 'train', processing_steps, params)
    generate_preprocessing_file(output_folder, 'test', processing_steps, params)

    return folder_name


if __name__ == "__main__":
    generate_preprocessed_data(['ContractionExpander', 'NumberUnifier', 'SpellingCorrector', 'MedicationRemover', 'SentenceTokenizer', 'WordTokenizer', 'PunctuationRemover', 'LowerCaseTransformer'])
