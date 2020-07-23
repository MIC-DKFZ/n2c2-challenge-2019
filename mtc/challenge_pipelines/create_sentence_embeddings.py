import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from mtc.core.embeddings import DocumentEmbeddings
from mtc.core.sentence import Sentence
#from mtc.settings import NLP_RAW_DATA


class SentenceEmbedder:
    def __init__(self):
        self.document_embedding = DocumentEmbeddings('InferSentEmbeddings', version=2)

    def get_embeddings(self, sentences):
        sentences = [Sentence(s) for s in sentences]
        self.document_embedding.embed_str(sentences)
        return [sentence.embedding.numpy() for sentence in sentences]


#folder = os.path.join(NLP_RAW_DATA, 'n2c2', 'preprocessed_data_2019-07-24_13-18-06')

networkdrive = '/home/klaus/networkdrives/'

relative_dir = 'E132-Projekte/Projects/2019_n2c2_challenge/submission_generation/03_12_2020_20_18_37_original_data/bert_base/1_example_run_step1/preprocessed_data_2020-03-15_21-49-55'

folder = os.path.join(networkdrive, relative_dir)
df_train = pd.read_csv(os.path.join(folder, 'preprocessed_data_train.tsv'), sep='\t', index_col='index')
df_test = pd.read_csv(os.path.join(folder, 'preprocessed_data_test.tsv'), sep='\t',  index_col='index')

df = pd.concat([df_train, df_test])
df_embeddings = df.copy()

embedder = SentenceEmbedder()

df_embeddings['sentence a'] = embedder.get_embeddings(df['sentence a'])
df_embeddings['sentence b'] = embedder.get_embeddings(df['sentence b'])


df_embeddings.to_pickle(os.path.join(folder, 'sentence_embeddings.pickle'))

vectors = np.array(df_embeddings['sentence a'].tolist() + df_embeddings['sentence b'].tolist())
projections = TSNE(n_components=2, random_state=1337).fit_transform(vectors)

np.save(os.path.join(folder, 'tsne_vectors'), vectors)
np.save(os.path.join(folder, 'tsne_projections'), projections)
