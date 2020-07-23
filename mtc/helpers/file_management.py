import os
import zlib

import pandas as pd

from mtc.settings import NLP_RAW_DATA


def load_sts_data(rel_n2c2_path):
    n2c2_path = os.path.join(NLP_RAW_DATA, rel_n2c2_path)

    # # Check if the correct file is loaded
    # with open(n2c2_path, 'rb') as file:
    #     hash = zlib.adler32(file.read())
    #
    # assert hash == 469557484, 'The checksum of the sts training data file is not correct. Please make sure that you are using the latest version.'

    sts_data = dict()
    df_data = pd.read_csv(n2c2_path, sep='\t', header=None)
    sts_data['raw_sentences_a'] = df_data[0].to_list()
    sts_data['raw_sentences_b'] = df_data[1].to_list()
    assert len(sts_data['raw_sentences_a']) == len(sts_data['raw_sentences_b']), 'Number of sentences do not match up.'

    if 2 in df_data.columns:
        sts_data['similarity_score'] = df_data[2].to_list()
        assert len(sts_data['raw_sentences_a']) == len(sts_data['similarity_score']), 'Each sentence must have a corresponding score'

    return sts_data


def len_sts_data(filename):
    df = pd.read_csv(os.path.join(NLP_RAW_DATA, 'n2c2', filename), sep='\t', header=None)
    
    return len(df)


def save_augmented_sts_data(sts_data, rel_n2c2_path):
    n2c2_path = os.path.join(NLP_RAW_DATA, rel_n2c2_path)
    df_data = pd.DataFrame(sts_data)
    df_data.to_csv(n2c2_path, sep='\t', header=None, index=False)
