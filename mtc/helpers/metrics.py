from scipy.stats import pearsonr


def pearson_score(*args, **kwargs):
    return pearsonr(*args, **kwargs)[0]
