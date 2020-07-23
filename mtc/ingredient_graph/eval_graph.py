from mtc.ingredient_graph.ingredients import add_ingredients
from mtc.settings import NLP_RAW_DATA
from mtc.ingredient_graph.graph_calculations import MedicalGraph
from mtc.ingredient_graph.tuple_calculations import add_tablet_diffs

import numpy as np
from sklearn.svm import SVR
import pandas as pd
import os
import glob
import re

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool


class ScalerProxy(object):
    def fit(self, data):
        pass

    def transform(self, data):
        return data


class MedicalGraphEstimator(object):
    def __init__(self, param):
        self.param = param

    def train(self, df_train):
        self.scaler_tablet = add_tablet_diffs(df_train)

        self.graph = MedicalGraph(self.param['factors'])
        self.graph.build(df_train)

        self._train_regressor(df_train)

    def predict(self, df_test):
        add_tablet_diffs(df_test, self.scaler_tablet)

        indices = []
        preds_final = []
        preds_graph = []

        for i, row in df_test.iterrows():
            if row['ingr_a'] is not None and row['ingr_b'] is not None:
                pred = self.graph.predict(row['ingr_a'], row['ingr_b'], row['tablet_diff'])
                if pred is not None:
                    preds_graph.append(pred)

                    feature = np.array([pred, row['input_score']], ndmin=2)

                    final_pred = self.regressor.predict(self.scaler_regressor.transform(feature))[0]
                    preds_final.append(final_pred)

                    indices.append(i)

        assert len(indices) == len(preds_final)
        return indices, preds_final, preds_graph

    def visualize_regression(self, preds_input, preds_graph, targets):
        x_min = 0
        x_max = 5
        y_min = 0
        y_max = 5
        XX, YY = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

        Z = self.regressor.predict(np.asarray([XX.ravel(), YY.ravel()]).transpose())
        ZZ = Z.reshape(XX.shape)

        cf = plt.contourf(XX, YY, ZZ, extend='both', levels=500, vmin=0, vmax=5)
        plt.colorbar(cf)
        plt.xlabel('graph scores')
        plt.ylabel('input scores')
        plt.scatter(preds_graph, preds_input, c=targets, cmap=cf.cmap, vmin=0, vmax=5)
        plt.clim(0, 5)
        plt.show()

        sns.jointplot(preds_graph, preds_input, kind='reg', joint_kws={'line_kws': {'color': 'red'}})
        plt.show()

        print()

    def _train_regressor(self, df):
        features = []
        labels = []

        for i, row in df.iterrows():
            if row['ingr_a'] is not None and row['ingr_b'] is not None:
                pred = self.graph.predict(row['ingr_a'], row['ingr_b'], row['tablet_diff'])
                if pred is not None:
                    features.append([pred, row['input_score']])
                    labels.append(row['score'])

        features = np.array(features)

        # scaler_regressor = MinMaxScaler()
        scaler_regressor = ScalerProxy()
        scaler_regressor.fit(features)

        regressor = SVR(gamma='auto', C=self.param['c'], epsilon=self.param['e'])
        regressor.fit(scaler_regressor.transform(features), labels)

        self.regressor = regressor
        self.scaler_regressor = scaler_regressor


def run_test_set(param, ingredients_folder, input_scores_folder, output_folder):
    # Data frames already contain ingredient information
    df_train = pd.read_csv(os.path.join(ingredients_folder, 'preprocessed_data_to_extract_ingredients', 'preprocessed_data_train_with_ingredients.tsv'), sep='\t')
    df_test = pd.read_csv(os.path.join(ingredients_folder, 'preprocessed_data_to_extract_ingredients', 'preprocessed_data_test_with_ingredients.tsv'), sep='\t')

    # Add input scores
    df_train['input_score'] = pd.read_csv(os.path.join(input_scores_folder, '0_dev_prediction.csv'), header=None)[0].tolist()
    df_test['input_score'] = pd.read_csv(os.path.join(input_scores_folder, '0_test_prediction.csv'), header=None)[0].tolist()

    estimator = MedicalGraphEstimator(param)
    estimator.train(df_train)
    indices, preds_final, _ = estimator.predict(df_test)

    df_test.loc[indices, 'input_score'] = preds_final
    df_test['input_score'].to_csv(os.path.join(output_folder, 'tablet_similarity_test.csv'), index=False, header=False)


def max_k_in_folder(folder):
    max_k = -1

    for folder in glob.glob(os.path.join(folder, 'kfold*')):
        match = re.search(r'kfold(\d+)', folder)
        assert match

        k = int(match.group(1))
        if k > max_k:
            max_k = k

    return max_k


def run_folds(param, ingredients_folder, input_scores_folder, output_folder):
    path_tab = os.path.join(ingredients_folder, 'preprocessed_data_to_extract_ingredients', 'preprocessed_data_train_with_ingredients.tsv')
    df_tab = pd.read_csv(path_tab, sep='\t')

    path_input_scores = os.path.join(input_scores_folder, '0_dev_prediction.csv')
    input_scores = np.array(pd.read_csv(path_input_scores, header=None)[0].tolist())

    assert len(input_scores) == len(df_tab)

    final_scores = pd.read_csv(os.path.join(input_scores_folder, '0_dev_prediction.csv'), header=None)

    estimator = MedicalGraphEstimator(param)

    all_targets = []
    all_preds_input = []
    all_preds_final = []

    max_k = max_k_in_folder(os.path.join(ingredients_folder, 'preprocessed_data_ingredients'))
    for idx in range(max_k + 1):
        df_train = pd.read_csv(os.path.join(ingredients_folder, 'preprocessed_data_ingredients', f'kfold{idx}', 'train.tsv'), sep='\t')
        df_test = pd.read_csv(os.path.join(ingredients_folder, 'preprocessed_data_ingredients', f'kfold{idx}', 'dev.tsv'), sep='\t')

        add_ingredients(df_tab, df_train)
        add_ingredients(df_tab, df_test)

        df_train['input_score'] = input_scores[df_train['index']]
        df_test['input_score'] = input_scores[df_test['index']]

        estimator.train(df_train)
        indices, preds_final, preds_graph = estimator.predict(df_test)

        # Find targets and generate final score list for this fold
        targets = []
        preds_input = []
        for idx, final_pred in zip(indices, preds_final):
            idx_sen = df_test.loc[idx, 'index']
            final_scores.loc[idx_sen, 0] = final_pred
            targets.append(df_test.loc[idx, 'score'])
            preds_input.append(df_test.loc[idx, 'input_score'])

        #estimator.visualize_regression(preds_input, preds_graph, targets)

        all_targets += targets
        all_preds_final += preds_final
        all_preds_input += preds_input

    final_scores.to_csv(os.path.join(output_folder, 'tablet_similarity_train.csv'), sep='\t', index=False, header=False)

    print(f'n = {len(all_targets)}, c = {param["c"]}, epsilon = {param["e"]}')
    mse_input = np.square(np.subtract(all_targets, all_preds_input)).mean()
    mse_graph = np.square(np.subtract(all_targets, all_preds_final)).mean()
    print(f'input: {mse_input}')
    print(f'graph: {mse_graph} (std = {np.std(all_preds_final)}, improvement per sentence = {(mse_input - mse_graph) / len(all_targets)})')

    # sns.jointplot(all_preds_input, all_targets, kind='reg', joint_kws={'line_kws': {'color': 'red'}})
    # plt.xlabel('Input')
    # plt.ylabel('Targets')
    # plt.xlim([0, 5])
    # plt.ylim([0, 5])
    # plt.show()
    # sns.jointplot(all_preds_final, all_targets, kind='reg', joint_kws={'line_kws': {'color': 'red'}})
    # plt.xlabel('Input + Graph')
    # plt.ylabel('Targets')
    # plt.xlim([0, 5])
    # plt.ylim([0, 5])
    # plt.show()

    return mse_graph


class ParameterEstimation(object):
    def __init__(self, param_init):
        self.param = param_init
        self.best_mse = run_folds(self.param)

    def optimize(self):
        for _ in range(2):
            self.random_walk()
            self.optimize_hyperparameters()

        print(f'best mse = {self.best_mse}, factors = {np.array2string(self.param["factors"], separator=",")}, c = {self.param["c"]}, e = {self.param["e"]}')

    def random_walk(self):
        for _ in range(50):
            i = np.random.randint(0, len(self.param['factors']))
            change = (1 * np.random.randn(1))[0]

            factors_current = self.param['factors'].copy()
            factors_current[i] += change

            print(f'testing with factors {factors_current}')
            mse = run_folds({'c': self.param['c'], 'e': self.param['e'], 'factors': factors_current})

            if mse < self.best_mse:
                self.best_mse = mse
                self.param['factors'] = factors_current
                print(f'Factors updated: best mse = {self.best_mse} with factors {np.array2string(factors_current, separator=",")}')

    def optimize_hyperparameters(self):
        params = []
        for e in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8]:
            for c in [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1, 2]:
                params.append({'c': c, 'e': e, 'factors': self.param['factors']})

        pool = Pool(6)
        result = pool.map(run_folds, params)
        pool.close()
        pool.join()

        for param, mse in zip(params, result):
            if mse < self.best_mse:
                self.best_mse = mse
                self.param = param
                print(f'Hyperparameters updated: best mse = {self.best_mse} with c = {param["c"]} and e = {param["e"]}')


def eval_graph(ingredients_folder, input_scores_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 0.5857391402015795
    param = {'c': 2, 'e': 0.7, 'factors': np.array([-4.03716946, 0.11850056, 3.30294487, 0.89707211, -2.12609579])}
    #param = {'c': 2, 'e': 0.7, 'factors': np.array([-0.86265979, 0.84737439, 8.57157767, 0.89707211, -1.22828961])}
    run_test_set(param, ingredients_folder, input_scores_folder, output_folder)
    run_folds(param, ingredients_folder, input_scores_folder, output_folder)


if __name__ == '__main__':
    folder = os.path.join(NLP_RAW_DATA, 'n2c2', 'preprocessed_data_2019-08-06_21-40-40_step3')
    eval_graph(folder, folder, folder)
    # param_estimator = ParameterEstimation(param)
    # param_estimator.optimize()
