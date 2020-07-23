import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import expit


class MedicalGraph(object):
    def __init__(self, factors, weight_attribute='score_scaled'):
        self.factors = factors
        self.weight_attribute = weight_attribute
        self.graph = None
        self.self_loop_mean = 0

    def build(self, df: pd.DataFrame):
        med_pairs = {}
        for i, row in df.iterrows():
            if not pd.isnull(row['ingr_a']) and not pd.isnull(row['ingr_b']):
                pair = tuple(sorted([row['ingr_a'], row['ingr_b']]))
                if pair not in med_pairs:
                    med_pairs[pair] = []

                med_pairs[pair].append({
                    'idx': i,
                    'score': df.loc[i, 'score'],
                    'tablet_diff': df.loc[i, 'tablet_diff'],
                    'input_score': row['input_score'],
                })

        edges = []
        for pair, values in med_pairs.items():
            scores = [v['score'] for v in values]
            scores_scaled = [self._add_graph_score(v['score'], v['tablet_diff']) for v in values]
            input_scores = [v['input_score'] for v in values]
            edges.append((pair[0], pair[1], {
                'score': np.mean(scores),
                'score_scaled': np.mean(scores_scaled),
                'input_score': np.mean(input_scores),
                'occurrences': len(scores)
            }))

        graph = nx.Graph()
        graph.add_edges_from(edges)

        # Count in how many sentences an ingredient occurs and add this information to the graph
        counts_a = df['ingr_a'].value_counts()
        counts_b = df['ingr_b'].value_counts()
        for node in graph.nodes:
            occurrences = 0
            if node in counts_a:
                occurrences += counts_a[node]
            if node in counts_b:
                occurrences += counts_b[node]
            assert occurrences > 0

            nx.set_node_attributes(graph, {node: occurrences}, 'occurrences')

        # The loop average is used when predicting ingredients to itself
        self.self_loop_mean = np.mean([graph.get_edge_data(start, end)[self.weight_attribute] for start, end in graph.edges if start == end])

        self.graph = graph

    def predict(self, ingr_a, ingr_b, tablet_diff):
        if self.graph.has_node(ingr_a) and self.graph.has_node(ingr_b) and nx.has_path(self.graph, source=ingr_a, target=ingr_b):
            s_path = nx.shortest_path(self.graph, source=ingr_a, target=ingr_b, weight=self.weight_attribute)
            weights = np.array(self._get_path_weights(s_path))
            if len(s_path) == 1:
                # Self-loops must be treated differently
                if self.graph.has_edge(s_path[0], s_path[0]):
                    pred = self.graph.get_edge_data(s_path[0], s_path[0])[self.weight_attribute]
                else:
                    # The loop does not exist in the train set. The best we can do is to take the average of all other loops
                    pred = self.self_loop_mean
            else:
                # pred = np.min(weights)
                # pred = np.median(weights)
                weights = weights[weights != 0]
                if weights.size == 0:
                    pred = 0
                else:
                    pred = 1 / np.sum(np.float_power(weights, -1))

            return self._get_graph_score(pred, tablet_diff)
        else:
            return None

    def _add_graph_score(self, current, diff):
        shift = np.tanh(np.sum(self.factors * diff))
        return np.clip(current + shift, 0, 5)

    def _get_graph_score(self, current, diff):
        shift = np.tanh(np.sum(self.factors * diff))
        return np.clip(current + shift, 0, 5)

    def _get_path_weights(self, path):
        start = path[0]
        weights = []
        for node in path[1:]:
            weights.append(self.graph.get_edge_data(start, node)[self.weight_attribute])
            start = node

        return weights
