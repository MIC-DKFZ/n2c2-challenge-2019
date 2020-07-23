# Feature reconstruction

## Step 1:

### Preprocessor
ContractionExpander
NumberUnifier
SpellingCorrector
SentenceTokenizer
WordTokenizer
PunctuationRemover
StopWordsRemover
LowerCaseTransformer
Lemmatizer

### Measures
Textdistance_Jaro_qval_3_
Textdistance_JaroWinkler_qval_3_
Textdistance_Sorensen_qval_3_
Textdistance_Overlap_qval_3_
Textdistance_Cosine_qval_3_
CosineSimilarity_
    -> DocumentPoolEmbeddings
        -> glove
CosineSimilarity_
    -> InferSentEmbeddings
        -> Version 2
VectorSimilarities_euclidean_
    -> DocumentPoolEmbeddings
        -> glove
VectorSimilarities_euclidean_
    -> InferSentEmbeddings
        -> Version 2
VectorSimilarities_manhattan_
    -> DocumentPoolEmbeddings
        -> glove
VectorSimilarities_manhattan_
    -> InferSentEmbeddings
        -> Version 2
VectorSimilarities_minkowski_
    -> DocumentPoolEmbeddings
        -> glove
VectorSimilarities_minkowski_
    -> InferSentEmbeddings
        -> Version 2
VectorSimilarities_cosine_similarity_
    -> DocumentPoolEmbeddings
        -> glove
VectorSimilarities_cosine_similarity_
    -> InferSentEmbeddings
        -> Version 2
        
## Step 2:

### Preprocessor
ContractionExpander
NumberUnifier
SpellingCorrector
SentenceTokenizer
WordTokenizer
PunctuationRemover
StopWordsRemover
LowerCaseTransformer
Lemmatizer

### Measures
BertResult_
Textdistance_DamerauLevenshtein_qval_3_
Textdistance_JaroWinkler_qval_3_
Textdistance_Bag_qval_3_
Textdistance_SqrtNCD_qval_3_
Textdistance_DamerauLevenshtein_qval_4_
Textdistance_JaroWinkler_qval_4_
Textdistance_Bag_qval_4_
Textdistance_SqrtNCD_qval_4_
VectorSimilarities_euclidean_
    -> DocumentPoolEmbeddings
        -> glove
VectorSimilarities_manhattan_
    -> DocumentPoolEmbeddings
        -> glove
VectorSimilarities_minkowski_
    -> DocumentPoolEmbeddings
        -> glove
VectorSimilarities_cosine_similarity_
    -> DocumentPoolEmbeddings
        -> glove
VectorSimilarities_euclidean_
    -> InferSentEmbeddings
        -> Version 1
VectorSimilarities_manhattan_
    -> InferSentEmbeddings
        -> Version 1
VectorSimilarities_minkowski_
    -> InferSentEmbeddings
        -> Version 1
VectorSimilarities_cosine_similarity_
    -> InferSentEmbeddings
        -> Version 1

### Voting regressors:
linear_model.LinearRegression(normalize=True, n_jobs= 29)
linear_model.Lasso(random_state=0, selection= 'cyclic',normalize=False)
linear_model.SGDRegressor(shuffle=True, penalty='l1', loss='squared_epsilon_insensitive', learning_rate='invscaling',
                               epsilon=0.1, early_stopping=False, average=True)
svm.LinearSVR(random_state=0, loss= 'squared_epsilon_insensitive', dual= True)
ensemble.RandomForestRegressor(max_depth=3, min_samples_split=2, random_state=0, n_estimators=700)
ensemble.AdaBoostRegressor(random_state=0, loss='linear', learning_rate=3.0, n_estimators=700)
ensemble.GradientBoostingRegressor()
