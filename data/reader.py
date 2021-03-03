import os
import csv
import json
from typing import List
from overrides import overrides
from data.instance import InputInstance

'''
DataReader: read data from file (including json and *.tsv)
'''
class DataReader(object):
    """Base class for data converters for sequence classification data sets."""
    NUM_LABELS = 0
    OUTPUT_MODE = 'classification'
    FILE_EXTENSION_TYPE = None

    def get_instances(self, data_dir: str, file_name: str):
        assert self.FILE_EXTENSION_TYPE is not None
        file_path = os.path.join(data_dir, '{}.{}'.format(file_name, self.FILE_EXTENSION_TYPE))
        if self.FILE_EXTENSION_TYPE == 'json':
            return self.create_instances(self._read_json(file_path), file_name)
        else:
            return self.create_instances(self._read_tsv(file_path), file_name)

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def create_instances(self, lines, data_type):
        raise NotImplementedError()

    def saving_instances(self, instances, data_dir, data_file):
        with open(os.path.join(data_dir, '{}.{}'.format(data_file, self.FILE_EXTENSION_TYPE)),'w', encoding='utf-8') as file:
            for instance in instances:
                file.write('{}\t{}\n'.format(instance.text_a, self.get_label_to_idx(instance.label)))

    def get_label_to_idx(self, label) -> int:
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        return label_map[label]

    def get_idx_to_label(self, label_idx) -> str:
        label_list = self.get_labels()
        id_to_label_map = {i: label for i, label in enumerate(label_list)}
        return id_to_label_map[label_idx]

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json list file."""
        with open(input_file, "r") as f:
            reader = f.readlines()
            lines = []
            for line in reader:
                lines.append(json.loads(line.strip()))
            return lines


class SnliDataReader(DataReader):
    """Reader for the Snli data set. """
    NUM_LABELS = 3
    FILE_EXTENSION_TYPE = 'tsv'

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    def create_instances(self, lines, set_type):
        """Creates instances for the training and dev sets."""
        instances = []
        for (i, line) in enumerate(lines):
            label, text_a, text_b = line
            instances.append(InputInstance(guid="{}-{}".format(set_type,i), text_a=text_a, text_b=text_b, label=label))
        return instances

    @overrides
    def saving_instances(self, instances: List[InputInstance], data_dir: str, data_file: str):
        with open(os.path.join(data_dir, '{}.tsv'.format(data_file)),'w', encoding='utf-8') as file:
            for instance in instances:
                file.write('{}\t{}\t{}\n'.format(instance.label, instance.text_a, instance.text_b))


class BinarySentimentAnalysisDataReader(DataReader):
    NUM_LABELS = 2
    FILE_EXTENSION_TYPE = 'txt'

    def get_labels(self):
        return ["100", "101"]

    def create_instances(self, lines, set_type):
        """Creates instances for the training and dev sets."""
        instances = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a, label = line
            label = "10" + label
            instances.append(InputInstance(guid=guid, text_a=text_a, text_b=None, label=label))
        return instances


class AgnewsReader(DataReader):
    NUM_LABELS = 4
    FILE_EXTENSION_TYPE = 'tsv'

    def get_labels(self):
        return ["100", "101", "102", "103"]

    def create_instances(self, lines, set_type):
        """Creates instances for the training and dev sets."""
        instances = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a, label = line
            label = "10" + label
            instances.append(InputInstance(guid=guid, text_a=text_a, text_b=None, label=label))
        return instances