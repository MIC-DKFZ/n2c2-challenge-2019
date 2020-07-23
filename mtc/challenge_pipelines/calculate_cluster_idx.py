import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
    

if __name__ == "__main__":


    NETWORK_SHARE = '/home/klaus/networkdrives/'
    # Results:
    SET_DIR = 'E132-Projekte/Projects/2019_n2c2_challenge/submission_generation/03_12_2020_20_18_37_original_data/bert_base/'
    TSNE_DIR = NETWORK_SHARE + SET_DIR + '1_example_run_step1/preprocessed_data_2020-03-15_21-49-55/'

    test_train_labels = 2 * (1642 * ['Training set'] + 412 * ['Test set'])
    train_idx_bool = [t == 'Training set' for t in test_train_labels]

    vectors = np.load(TSNE_DIR + 'tsne_vectors.npy')
    projections = np.load(TSNE_DIR + 'tsne_projections.npy')

    k = 10
    kmeans = KMeans(n_clusters=k, random_state=1337).fit(vectors[train_idx_bool])

    print('Quality of cluster', set(kmeans.labels_[0:1642]-kmeans.labels_[0:1642]))

    test_cluster = [0, 3, 4, 7, 9]
    ix = np.isin(kmeans.labels_[0:1642], test_cluster)
    np.save(os.path.join(TSNE_DIR, 'test_cluster_idx'), ix)