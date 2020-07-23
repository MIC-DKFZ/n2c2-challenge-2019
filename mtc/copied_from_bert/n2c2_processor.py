import os

from collections import Iterable
from mtc.copied_from_bert.utils_glue import InputExample, DataProcessor, processors, output_modes, pearson_and_spearman


class N2C2Processor(DataProcessor):
    """Processor for the n2c2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            if set_type != 'test':
                label = line[3]
            else:
                label = i - 1
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def compute_metrics(task_name, preds, labels):
    # Make sure we have at least a vector
    if len(preds.shape) == 0:
        preds = preds.reshape(-1)
    if len(labels.shape) == 0:
        labels = labels.reshape(-1)

    assert len(preds) == len(labels)
    assert task_name == "n2c2"

    if len(preds) == 1:
        return {
            "pearson": 0,
            "spearmanr": 0,
            "corr": 0,
        }

    return pearson_and_spearman(preds, labels)


processors['n2c2'] = N2C2Processor
output_modes['n2c2'] = 'regression'
