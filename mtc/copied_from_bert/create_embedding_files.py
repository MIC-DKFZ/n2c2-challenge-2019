import os

import pandas as pd
import torch
from pytorch_transformers import BertConfig, BertModel, BertTokenizer

from mtc.settings import NLP_RAW_DATA, NLP_EXPERIMENT_PATH
from mtc.mtc_properties import KFOLDS

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class BertEmbeddingsFineTuned:
    def __init__(self, folder):
        config = BertConfig.from_pretrained(os.path.join(folder, 'config.json'))
        self.tokenizer = BertTokenizer.from_pretrained(folder)
        self.model = BertModel(config).cuda()

    def get_embeddings(self, sentence, output_level):
        assert output_level == 0 or output_level == 1, 'output_level must either be 0 (word embeddings), or 1 (sentence embeddings)'

        input_ids = torch.tensor(self.tokenizer.encode(sentence)).unsqueeze(0).cuda()  # Batch size 1
        outputs = self.model(input_ids)
        last_hidden_layer = outputs[output_level]  # The last hidden-state is the first element of the output tuple
        return last_hidden_layer[0].cpu().detach().numpy()  # We have only one batch


def convert_file(file, embedder, output_level):
    data = pd.read_csv(file, sep='\t')
    embeddings = data.copy()

    for idx in range(len(embeddings)):
        embeddings.at[idx, 'sentence a'] = embedder.get_embeddings(embeddings.loc[idx, 'sentence a'], output_level)
        embeddings.at[idx, 'sentence b'] = embedder.get_embeddings(embeddings.loc[idx, 'sentence b'], output_level)

        progress = (idx + 1) / len(embeddings)
        if idx == len(embeddings) - 1:
            print('%.2f' % progress)
        else:
            print('%.2f' % progress, end='\r')

    return embeddings


experiment_name = 'preprocessed_data_2019-07-23_13-56-39'
input_dir = os.path.join(NLP_RAW_DATA, 'n2c2', experiment_name)
output_dir = os.path.join(NLP_EXPERIMENT_PATH, 'n2c2', experiment_name + '_biobert_pretrain_output_all_notes_150000')
output_level = 1

for fold_idx in range(KFOLDS):
    kfold_folder = 'kfold{idx}'.format(idx=fold_idx)
    embedder = BertEmbeddingsFineTuned(os.path.join(output_dir, kfold_folder))
    print('# fold {}'.format(fold_idx))

    for data_file in ['train', 'dev']:
        embeddings = convert_file(os.path.join(input_dir, kfold_folder, data_file + '.tsv'), embedder, output_level)
        embeddings.to_pickle(os.path.join(output_dir, kfold_folder, data_file + str(output_level) + '.pickle'))

print('All tasks completed')
