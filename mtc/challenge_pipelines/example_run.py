import os
from datetime import datetime
from itertools import chain, combinations


from sklearn import linear_model, ensemble, svm, gaussian_process, neural_network
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.base import clone
from scipy.stats import pearsonr
from scipy.stats import randint as sp_randint

import pandas as pd
import numpy as np

from mtc.helpers.file_management import load_sts_data
from mtc.helpers.metrics import pearson_score
from mtc.helpers.text_processing import TextProcessing
from mtc.helpers.util import get_ingredient_booleans
from mtc.helpers.util import PipelineDictArgument
from mtc.core.evaluator import Evaluator
from mtc.core.similarity_measures import SimilarityMeasures
from mtc.core.experiment import Measuring, Training, Predicting, Evaluating
from mtc.settings import NLP_EXPERIMENT_PATH, NLP_MODELS_PATH

def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

def measure_dataset(measure_config, pickle_folder, sts_data_path, train_or_test=None):
    measuring = Measuring(measure_config, pickle_folder)
    measuring.set_sts_data_dict(sts_data_path)
    measuring.measure(train_or_test)
    measuring.save_sentences_objects()
    measuring.create_features()
    measuring.save_feature_matrix()
    measuring.save_measuring_config()
    #measuring.plot_correlation_matrix()
    return measuring


########################################################################################################################
# Preprocessors
########################################################################################################################

pa_preprocessor1 = PipelineDictArgument('SelectivePreprocessor', [
        PipelineDictArgument('ContractionExpander'),
        PipelineDictArgument('NumberUnifier'),
        PipelineDictArgument('SpellingCorrector'),
        PipelineDictArgument('SentenceTokenizer'),
        PipelineDictArgument('WordTokenizer'),
        PipelineDictArgument('PunctuationRemover'),
        PipelineDictArgument('StopWordsRemover'),
        PipelineDictArgument('LowerCaseTransformer'),
        PipelineDictArgument('Lemmatizer'),
    ])

pa_preprocessor2 = PipelineDictArgument('SelectivePreprocessor', [
        PipelineDictArgument('NumberUnifier'),
        PipelineDictArgument('SpellingCorrector'),
        PipelineDictArgument('SentenceTokenizer'),
        PipelineDictArgument('WordTokenizer'),
        PipelineDictArgument('LowerCaseTransformer'),
    ])

# pa_preprocessor2 = PipelineDictArgument('SelectivePreprocessor', [
#         #PipelineDictArgument('RemoveActiveIngredients'),
#         PipelineDictArgument('NumberUnifier'),
#         PipelineDictArgument('SpellingCorrector'),
#         PipelineDictArgument('SentenceTokenizer'),
#         PipelineDictArgument('WordTokenizer'),
#         PipelineDictArgument('PunctuationRemover'),
#         PipelineDictArgument('LowerCaseTransformer'),
#         PipelineDictArgument('RemoveHighNGrams', 10, 6),
#         PipelineDictArgument('StopWordsRemover'),
#         PipelineDictArgument('Lemmatizer'),
#     ])

pa_token_embeddings = [
    # Need RemoveLowHighFrequencyWords
    #PipelineDictArgument('WordEmbeddings', 'en'),
    PipelineDictArgument('WordEmbeddings', 'glove'),
    #PipelineDictArgument('WordEmbeddings', os.path.join(NLP_MODELS_PATH, 'pretrained', 'word_embeddings', 'pubmed2018_w2v_400D', 'pubmed2018_w2v_400D.bin'),),
]

pa_document_embeddings = [
   # PipelineDictArgument('DocumentPoolEmbeddings', pa_token_embeddings),
    #PipelineDictArgument('DocumentPoolEmbeddings', pa_token_embeddings, pooling='min'),
    #PipelineDictArgument('DocumentPoolEmbeddings', pa_token_embeddings, pooling='max'),
    #PipelineDictArgument('InferSentEmbeddings', version=1),
   # PipelineDictArgument('InferSentEmbeddings', version=2),
    #PipelineDictArgument('GoogleUseEmbeddings')
    #PipelineDictArgument('Doc2VecEmbeddings',  os.path.join(NLP_MODELS_PATH, 'trained', 'document_embeddings',
                                                            #'doc2vec_trained', 'doc2vec_trained.model'))
]


########################################################################################################################
# Similarity measures
########################################################################################################################

pooled_sentence_measures = [SimilarityMeasures('VectorSimilarities', pa_preprocessor1, PipelineDictArgument('DocumentPoolEmbeddings', pa_token_embeddings), name)
                     for name in ['euclidean', 'manhattan', 'minkowski', 'cosine_similarity']]

infersent_1_sentence_measures = [SimilarityMeasures('VectorSimilarities', pa_preprocessor1, PipelineDictArgument('InferSentEmbeddings', version=1), name)
                     for name in ['euclidean', 'manhattan', 'minkowski', 'cosine_similarity']]

infersent_2_sentence_measures = [SimilarityMeasures('VectorSimilarities', pa_preprocessor1, PipelineDictArgument('InferSentEmbeddings', version=2), name)
                     for name in ['euclidean', 'manhattan', 'minkowski', 'cosine_similarity']]

google_use_sentence_measures = [SimilarityMeasures('VectorSimilarities', pa_preprocessor1, PipelineDictArgument('GoogleUseEmbeddings'), name)
                     for name in ['euclidean', 'manhattan', 'minkowski', 'cosine_similarity']]

bert_sentence_measures = [SimilarityMeasures('VectorSimilarities', pa_preprocessor2, PipelineDictArgument('BertSentenceEmbeddings'), name)
                     for name in ['euclidean', 'manhattan', 'minkowski', 'cosine_similarity']]

glove_document_pool_sentence_difference = SimilarityMeasures('VectorSimilarities', pa_preprocessor1, PipelineDictArgument('DocumentPoolEmbeddings', [PipelineDictArgument('WordEmbeddings', 'glove')]), 'cosine_similarity')

token_measures = [SimilarityMeasures('WMDDistance', pa_preprocessor1, token_embedding) for token_embedding in pa_token_embeddings]

#includes_medication_similarity = SimilarityMeasures('IncludesMedicationSimilarity', pa_preprocessor2)
tablet_similarity = SimilarityMeasures('TabletFeatures', pa_preprocessor1)

# For bert Jaro, JaroWinkler, Sorensen, Overlap, Cosine,
textdistance_names = [
    #'Hamming',  # bad performance
    'DamerauLevenshtein', # performce okay
    #'Levenshtein', # performce okay
    #'Mlipns', # todo: does not work
    #'Jaro',
    'JaroWinkler',
    #'NeedlemanWunsch', # performce okay, takes forever
    #'Gotoken_measures1toh',# takes forever
    #'SmithWaterman', # performce okay, takes forever

    # Token based
    #'Jaccard', # performce okay
    #'Sorensen',
    # 'Tversky', # performs really bad?!
    #'Overlap',
    #'Tanimoto',# todo: does not work
    #'Cosine',
    #'MongeElkan', # bad performance
    'Bag',

    # Sequence based
    #'LCSSeq', # todo: does not work
    #'LCSStr',# todo: does not work
    #'RatcliffObershelp', # todo: does not work

    # Compression based
    #'ArithNCD',# todo: does not work
    #'RLENCD',# todo: does not work
    # 'BWTRLENCD', I don't know...
    'SqrtNCD',
    #'EntropyNCD',

    # # Simple -> are very bad
    # 'Prefix',
    # #'Postfix', # todo: does not work
    # 'Length',
    # 'Identity',
    # 'Matrix',
]

#measures_textdistance1 = [SimilarityMeasures('Textdistance', pa_preprocessor1, measure, qval=1) for measure in textdistance_names]
measures_textdistance2 = [SimilarityMeasures('Textdistance', pa_preprocessor1, measure, qval=2) for measure in textdistance_names]
measures_textdistance3 = [SimilarityMeasures('Textdistance', pa_preprocessor1, measure, qval=3) for measure in textdistance_names]
measures_textdistance4 = [SimilarityMeasures('Textdistance', pa_preprocessor1, measure, qval=4) for measure in textdistance_names]
measures_textdistance5 = [SimilarityMeasures('Textdistance', pa_preprocessor1, measure, qval=5) for measure in textdistance_names]
textdistance_measures = measures_textdistance2 + measures_textdistance3 + measures_textdistance4

relevant_textdistance_measures = [
    SimilarityMeasures('Textdistance', pa_preprocessor1, 'DamerauLevenshtein', qval=3),
    SimilarityMeasures('Textdistance', pa_preprocessor1, 'DamerauLevenshtein', qval=4),
    SimilarityMeasures('Textdistance', pa_preprocessor1, 'Jaro', qval=3),
    SimilarityMeasures('Textdistance', pa_preprocessor1, 'Jaro', qval=4),
    SimilarityMeasures('Textdistance', pa_preprocessor1, 'SqrtNCD', qval=3),
    SimilarityMeasures('Textdistance', pa_preprocessor1, 'SqrtNCD', qval=4),
    SimilarityMeasures('Textdistance', pa_preprocessor1, 'Bag', qval=3),
    SimilarityMeasures('Textdistance', pa_preprocessor1, 'Bag', qval=4),
]

document_embedding_as_feature = SimilarityMeasures('AddDocumentEmbedding', pa_preprocessor1, PipelineDictArgument('InferSentEmbeddings', version=2))

measure_bert = [SimilarityMeasures('BertResult', pa_preprocessor1)]
medication_graph = [SimilarityMeasures('MedicationGraph', pa_preprocessor1)]
sequence_matcher_similarity1 = SimilarityMeasures('SequenceMatcherSimilarity', pa_preprocessor1)
#sequence_matcher_similarity2 = SimilarityMeasures('SequenceMatcherSimilarity', pa_preprocessor2)


########################################################################################################################
# Estimator
########################################################################################################################

r1 = linear_model.LinearRegression(normalize=True, n_jobs= 29)
r2 = ensemble.RandomForestRegressor(max_depth=3, min_samples_split=2, random_state=0, n_estimators=700)
r3 = ensemble.AdaBoostRegressor(random_state=0, loss='linear', learning_rate=3.0, n_estimators=700)
r4 = ensemble.GradientBoostingRegressor()
r5 = ensemble.BaggingRegressor() # overfitting
r6 = ensemble.ExtraTreesRegressor() # overfitting
r7 = linear_model.BayesianRidge(normalize=True)
r8 = linear_model.ARDRegression(normalize=True)
r9 = linear_model.HuberRegressor()
r10 = linear_model.Lasso(random_state=0, selection= 'cyclic',normalize=False)
r11 = svm.LinearSVR(random_state=0, loss= 'squared_epsilon_insensitive', dual= True)
r12 = gaussian_process.GaussianProcessRegressor() # overfitting
r13 = linear_model.PassiveAggressiveRegressor() # takes okayisch time
r14 = linear_model.RANSACRegressor() # overfitting?
r15 = linear_model.SGDRegressor(shuffle=True, penalty='l1', loss='squared_epsilon_insensitive', learning_rate='invscaling',
                               epsilon=0.1, early_stopping=False, average=True)
r16 = linear_model.TheilSenRegressor() # eher Verschlechterung
# r17 = neural_network.MLPRegressor()

# #Unoptimized
# r1 = linear_model.LinearRegression()
# r2 = ensemble.RandomForestRegressor(max_depth=3, min_samples_split=2, random_state=0, n_estimators=700)
# r3 = ensemble.AdaBoostRegressor(random_state=0, n_estimators=100)
# r4 = ensemble.GradientBoostingRegressor()
# r5 = ensemble.BaggingRegressor() # overfitting
# r6 = ensemble.ExtraTreesRegressor() # overfitting
# r7 = linear_model.BayesianRidge()
# r8 = linear_model.ARDRegression()
# r9 = linear_model.HuberRegressor()
# r10 = linear_model.Lasso()
# r11 = svm.LinearSVR()
# r12 = gaussian_process.GaussianProcessRegressor() # overfitting
# r13 = linear_model.PassiveAggressiveRegressor() # takes okayisch time
# r14 = linear_model.RANSACRegressor() # overfitting?
# r15 = linear_model.SGDRegressor()
# r16 = linear_model.TheilSenRegressor() # eher Verschlechterung

rs = [r1, r2, r3, r4, r10, r11, r15]
regressor_list = []
for idx, r in enumerate(rs):
    regressor_list.append((f'r{idx}', r))
estimator = ensemble.VotingRegressor(regressor_list)
#estimator = ensemble.RandomForestRegressor(max_depth=3, min_samples_split=2, random_state=0, n_estimators=700)

# # 37 features
# estimator_optimization= [
#     {
#         'estimator':  ensemble.RandomForestRegressor(random_state=0),
#         'params_dist': {"max_depth": [3, None],
#               "max_features": sp_randint(1, 38),
#               "min_samples_split": sp_randint(2, 11),
#               "bootstrap": [True, False],
#               "n_estimators": [100, 700]}
#     },
# ]
#
# estimator_list = [RandomizedSearchCV(rand_search_cv_config['estimator'], param_distributions=rand_search_cv_config['params_dist'], n_iter=10, cv=5, iid=False)
#                   for rand_search_cv_config in estimator_optimization]

#estimator = ensemble.RandomForestRegressor(max_depth=3, min_samples_split=2, random_state=0, n_estimators=700)
#estimator = ensemble.RandomForestRegressor(max_depth=3, min_samples_split=2, random_state=0, n_estimators=700)
#estimator = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, scoring='precision')
#param_grid = {'n_estimators': [100, 200, 700], 'max_depth': [None], 'min_samples_split': [2]}
#estimator = GridSearchCV(ensemble.RandomForestRegressor(), param_grid, cv=5, scoring=make_scorer(pearson_score))


def create_features_for_bert(sts_data_path, train_or_test_folder, time_stamp): # train_or_test_folder either: 'train' or 'test'

    measure_config_list = [
        {
            'measures': relevant_textdistance_measures,
        },
        {
             'measures': measures_textdistance3,
        },
        {
             'measures': measures_textdistance3 + document_measures,
        },
        # {
        #      'measures': [medication_graph] + measures_textdistance3 + document_measures,
        # }
    ]

    for exp_idx, measure_config in enumerate(measure_config_list):
        pickle_folder = os.path.join(NLP_EXPERIMENT_PATH, 'pickles_for_bert', train_or_test_folder, f'{time_stamp}_{exp_idx}')
        os.makedirs(pickle_folder)
        measure_dataset(measure_config['measures'], pickle_folder, sts_data_path)


def optimizing_bert_results(sts_data_train_path, sts_data_test_path, timestamp):

    pickle_folder_train = os.path.join(NLP_EXPERIMENT_PATH, 'pickles_for_opt', 'train', f'{timestamp}')
    pickle_folder_test = os.path.join(NLP_EXPERIMENT_PATH, 'pickles_for_opt', 'test', f'{timestamp}')

    if not os.path.isdir(pickle_folder_train):
        os.mkdir(pickle_folder_train)
    if not os.path.isdir(pickle_folder_test):
        os.mkdir(pickle_folder_test)

    all_measures_list = [medication_graph,
                         measures_textdistance3 + measures_textdistance4,
                         token_measures,
                         pooled_sentence_measures,
                         infersent_1_sentence_measures,
                         infersent_2_sentence_measures,
                         google_use_sentence_measures
                         ]

    measure_combinations = []
    for subset in all_subsets(all_measures_list):
        measure_list = []
        for sub in subset:
            measure_list = measure_list + sub
        measure_combinations.append(measure_list)

    measure_combinations = measure_combinations[1:]

    print(len(measure_combinations))
    experiments_schedule = [
        # {
        #     'measures': measure_bert + measure_combi,
        #     'estimator': estimator,
        # } for measure_combi in measure_combinations
        # {
        #     'measures': measure_combi,
        #     'estimator': estimator,
        # } for measure_combi in all_measures_list
        {
            'measures': measure_bert + medication_graph + measures_textdistance3 + measures_textdistance4 + pooled_sentence_measures + infersent_1_sentence_measures,
            'estimator': estimator,
        },
        # {
        #     'measures': [measure_bert] + relevant_textdistance_measures,
        #     'estimator': estimator,
        # },
        # {
        #     'measures': [measure_bert] + relevant_textdistance_measures + document_measures,
        #     'estimator': estimator,
        # },
        # {
        #     'measures': [measure_bert, medication_graph] + relevant_textdistance_measures,
        #     'estimator': estimator,
        # },
    ]

    kf = KFold(n_splits=150)

    for exp_idx, experiment_config in enumerate(experiments_schedule):

        # Train dataset
        measuring_train = measure_dataset(experiment_config['measures'], pickle_folder_train, sts_data_train_path,
                                          'train')
        X_train, y_train, raw_sentences_a_train, raw_sentences_b_train = measuring_train()

        # Test Dataset
        measuring_test = measure_dataset(experiment_config['measures'], pickle_folder_test, sts_data_test_path, 'test')
        X_test, y_test, raw_sentences_a_test, raw_sentences_b_test = measuring_test()

        y_fold_dev_predicted_list = []
        y_test_predicted_list = []

        # Applying a k-Fold
        for idx, (train_index, dev_index) in enumerate(kf.split(X_train)):
            print(f'kfold {idx}')
            X_fold_train, X_fold_dev = X_train[train_index], X_train[dev_index]
            y_fold_train, y_fold_dev = y_train[train_index], y_train[dev_index]

            # Init estimator
            est = clone(experiment_config['estimator'])

            # Apply training
            training = Training(est)
            training(X_fold_train, y_fold_train)

            # Predict on test/dev data
            predicting = Predicting(est)
            y_fold_dev_predicted = predicting(X_fold_dev)
            y_fold_dev_predicted_list = y_fold_dev_predicted_list + list(y_fold_dev_predicted)
            print(f'Dev pearson {pearsonr(y_fold_dev_predicted, y_fold_dev)}')

            y_fold_train_predicted = predicting(X_fold_train)
            print(f'Train pearson {pearsonr(y_fold_train_predicted, y_fold_train)}')

            # Predict on test/dev data
            predicting = Predicting(est)
            y_test_predicted = predicting(X_test)
            y_test_predicted_list.append(np.array(y_test_predicted))

        pearson_filename = os.path.join(NLP_EXPERIMENT_PATH, f"averaged_pcc .csv")
        total_dev_pearson = pearsonr(y_fold_dev_predicted_list, y_train)[0]
        print(f"pearson {total_dev_pearson}")

        pearson_list_dict = {'Average pearson': [total_dev_pearson], 'Timestamp': timestamp, 'Rund idx': exp_idx}
        df_pearson = pd.DataFrame(pearson_list_dict)
        if not os.path.isfile(pearson_filename):
            df_pearson.to_csv(pearson_filename, header=list(df_pearson.columns), index=False)
        else:  # else it exists so append without writing the header
            df_pearson.to_csv(pearson_filename,  mode='a', header=False, index=False)

        final_prediction = pd.DataFrame({'scores': y_fold_dev_predicted_list})
        final_prediction.to_csv(os.path.join(NLP_EXPERIMENT_PATH, f"{timestamp}_{exp_idx}_dev_prediction.csv"),
                                header=False, index=False)

        y_test_predicted_list = np.array(y_test_predicted_list)
        y_final_test_prediction = y_test_predicted_list.mean(axis=0)
        final_prediction = pd.DataFrame({'scores': y_final_test_prediction})
        final_prediction.to_csv(os.path.join(NLP_EXPERIMENT_PATH, f"{timestamp}_{exp_idx}_test_prediction.csv"),
                                header=False, index=False)


if __name__ == '__main__':

    sts_data_train_path = os.path.join('n2c2', 'clinicalSTS2019.train.txt')
    sts_data_test_path = os.path.join('n2c2', 'clinicalSTS2019.test.txt')

    time_stamp = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')

    # # To create for bert data sets
    # create_features_for_bert(sts_data_train_path, 'train', time_stamp)
    # create_features_for_bert(sts_data_test_path, 'test', time_stamp)

    #time_stamp = '08_06_2019_19_20_36'
    # To apply post optimization

    time_stamp = 'some_features'
    optimizing_bert_results(sts_data_train_path, sts_data_test_path, time_stamp)
