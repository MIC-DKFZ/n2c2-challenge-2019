import argparse
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from mtc.challenge_pipelines.preprocess_data import generate_preprocessed_data
from mtc.settings import NLP_EXPERIMENT_PATH, NLP_RAW_DATA
from mtc.core.embeddings import DocumentEmbeddings
from mtc.core.sentence import Sentence


root_path = Path(NLP_RAW_DATA) / 'n2c2'
train_name = 'clinicalSTS2019.train.txt'
test_name = 'clinicalSTS2019.test.txt'
test_labels_name = 'clinicalSTS2019.test.gs.sim.txt'


def prepare_input_folder_official(folder):
    folder = folder / 'n2c2'
    folder.mkdir(parents=True, exist_ok=True)

    # Just copy the original data
    shutil.copy2(root_path / train_name, folder / train_name)
    shutil.copy2(root_path / test_name, folder / test_name)
    shutil.copy2(root_path / test_labels_name, folder / test_labels_name)


def prepare_input_folder_complete(folder):
    folder = folder / 'n2c2'
    folder.mkdir(parents=True, exist_ok=True)

    # Load raw data
    df_train = pd.read_csv(root_path / train_name, sep='\t', header=None)
    df_test = pd.read_csv(root_path / test_name, sep='\t', header=None)
    df_test[2] = pd.read_csv(root_path / 'clinicalSTS2019.test.gs.sim.txt', header=None)

    # Create a combined dataframe
    df = pd.concat([df_train, df_test])

    # Shuffle the rows
    df = df.sample(frac=1, random_state=1337).reset_index(drop=True)

    # Use combined file as new training data
    df.to_csv(folder / train_name, sep='\t', header=None, index=None)

    # Check whether the new files has the correct format
    df_train = pd.read_csv(folder / train_name, sep='\t', header=None)
    assert df_train.shape == (2054, 3)
    assert pd.api.types.is_object_dtype(df.dtypes[0])
    assert pd.api.types.is_object_dtype(df.dtypes[1])
    assert pd.api.types.is_float_dtype(df.dtypes[2])

    # Copy the test as-is so that the algorithm won't break (however, the results for the test set are not useful)
    shutil.copy2(root_path / test_name, folder / test_name)
    shutil.copy2(root_path / test_labels_name, folder / test_labels_name)


def prepare_input_folder_cluster(folder):
    folder = folder / 'n2c2'
    folder.mkdir(parents=True, exist_ok=True)

    # Apply basic preprocessing for the InferSent embeddings
    preprocessing_name = generate_preprocessed_data(['ContractionExpander', 'NumberUnifier', 'SpellingCorrector', 'LowerCaseTransformer'], output_folder=folder)
    df_train = pd.read_csv(folder / preprocessing_name / 'preprocessed_data_train.tsv', sep='\t', index_col='index')
    df_test = pd.read_csv(folder / preprocessing_name / 'preprocessed_data_test.tsv', sep='\t',  index_col='index')

    # Calculate the InferSent embeddings
    df = pd.concat([df_train, df_test])
    df_embeddings = df.copy()

    class SentenceEmbedder:
        def __init__(self):
            self.document_embedding = DocumentEmbeddings('InferSentEmbeddings', version=2)

        def get_embeddings(self, sentences):
            sentences = [Sentence(s) for s in sentences]
            self.document_embedding.embed_str(sentences)

            return [sentence.embedding.numpy() for sentence in sentences]
    
    embedder = SentenceEmbedder()
    df_embeddings['sentence a'] = embedder.get_embeddings(df['sentence a'])
    df_embeddings['sentence b'] = embedder.get_embeddings(df['sentence b'])

    # Calculate kmeans clustering
    vectors = np.array(df_embeddings['sentence a'].tolist() + df_embeddings['sentence b'].tolist())

    test_train_labels = 2 * (1642 * ['Training set'] + 412 * ['Test set'])
    train_idx_bool = [t == 'Training set' for t in test_train_labels]
    kmeans = KMeans(n_clusters=10, random_state=1337).fit(vectors[train_idx_bool])

    # Select a subset of clusters used for training
    test_cluster = [0, 3, 4, 7, 9]  # cf. PaperFigures.ipynb notebook for more information why these clusters were selected
    ix = np.isin(kmeans.labels_[0:1642], test_cluster)

    # Create the corresponding train data set
    train_data = pd.read_csv(root_path / train_name, sep='\t', header=None)
    train_data = train_data[ix]
    train_data.to_csv(folder / train_name, sep='\t', header=None, index=False)

    # Test data remains unchainged
    shutil.copy2(root_path / test_name, folder / test_name)
    shutil.copy2(root_path / test_labels_name, folder / test_labels_name)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--split', required=True, choices=['official', 'complete', 'cluster'], default='official', help='The dataset split to use (official=train and test set, complete=combined test and train set, cluster=subset of the training data which sentence clusters being also present in the test data).')
args = parser.parse_args()

# Create a folder for the complete run
time_stamp = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
base_folder = Path(NLP_EXPERIMENT_PATH) / 'submission_generation' / time_stamp
input_folder = base_folder / 'input_data'

# Create the input data used for this run
if args.split == 'official':
    prepare_input_folder_official(input_folder)
elif args.split == 'complete':
    prepare_input_folder_complete(input_folder)
elif args.split == 'cluster':
    prepare_input_folder_cluster(input_folder)
else:
    print('Invalid dataset split')
    exit(1)

# Run the result generation while pointing to the new location of the input data
env = os.environ.copy()
env['NLP_RAW_DATA'] = input_folder
env['BASE_FOLDER'] = base_folder
subprocess.call(f'python run_approaches.py', shell=True, env=env)
