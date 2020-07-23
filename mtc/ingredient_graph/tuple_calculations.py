import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def extract_tuple(sentence):
    def replace_range(match):
        num1 = match.group(1)
        num2 = match.group(2)

        return '{:.2f}'.format((float(num1) + float(num2)) / 2)

    # Replace ranges
    sentence = re.sub(r'(\d+\.?\d*|\d*\.?\d+)\s*-\s*(\d+\.?\d*|\d*\.?\d+)', replace_range, sentence)

    amount = 0
    unit_type = 0
    match = re.search(r'(\d+\.?\d*|\d*\.?\d+)\s*(mcg|mg|grams?|g|ml|liters?)\b', sentence)
    if match:
        number = float(match.group(1))
        unit = match.group(2)

        amount = number
        if unit == 'mg':
            unit_type = 1
        elif unit == 'mcg':
            unit_type = 2
        elif unit == 'g' or unit == 'gram' or unit == 'grams':
            unit_type = 3
        elif unit == 'ml':
            unit_type = 4
        elif unit == 'liter' or unit == 'liters':
            unit_type = 5

        # if unit == 'mg':
        #     amount = number
        # elif unit == 'mcg':
        #     amount = number / 1000
        # else:
        #     # gram
        #     amount = number * 1000

    frequency = 0
    tab_type = 0
    match = re.search(r'(\d+\.?\d*|\d*\.?\d+)\s+(tablet|capsule|package|puff)s?', sentence)
    if match:
        frequency = float(match.group(1))
        package = match.group(2)

        if package == 'tablet':
            tab_type = 1
        elif package == 'capsule':
            tab_type = 2
        elif package == 'package':
            tab_type = 3
        elif package == 'puff':
            tab_type = 4

    dose = 0  # In number of times per day
    match1 = re.search(r'(\d+\.?\d*|\d*\.?\d+)\s+(times?\s*(?:daily)?|hours?)', sentence)
    match2 = re.search(r'every\s+(?:bedtime|evening)', sentence)
    if match1:
        number = float(match1.group(1))
        unit = match1.group(2)

        if 'hour' in unit and number != 0:
            # E.g. every 4 hours --> 3 times per day (assuming that tablet are not supposed to be taken at night)
            dose = 12 / number
        else:
            dose = number
    elif match2:
        dose = 1

    return amount, unit_type, frequency, tab_type, dose


def tuple_diff(t1, t2):
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)

    assert len(t1) == len(t2)
    n_diff_features = len(t1)

    diff = np.zeros(n_diff_features)
    diff_types = ['ratio', 'nominal', 'ratio', 'nominal', 'ratio']

    for i, diff_type in zip(range(n_diff_features), diff_types):
        if diff_type == 'ratio':
            diff[i] = np.square(t1[i] - t2[i])
        elif diff_type == 'nominal':
            diff[i] = t1[i] != t2[i]

    return diff


def add_tablet_diffs(df, scaler=None):
    diffs = []
    indices = []
    for i, row in df.iterrows():
        if row['ingr_a'] is not None and row['ingr_b'] is not None:
            diff = tuple_diff(extract_tuple(df.loc[i, 'sentence a']), extract_tuple(df.loc[i, 'sentence b']))
            diffs.append(diff)
            indices.append(i)

    diffs = np.asarray(diffs)
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(diffs)

    diffs = scaler.transform(diffs)

    df['tablet_diff'] = [[] for _ in range(len(df))]
    for i, diff in zip(indices, diffs):
        df.at[i, 'tablet_diff'] = diff

    return scaler
