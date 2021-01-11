from torch.nn import MSELoss, CrossEntropyLoss
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer
from transformers import ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer
from transformers import AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from utils.metrics import ClassificationMetric
from data.reader import *


class PRETRAINED_MODEL_TYPE:
    MODEL_CLASSES = {
        # Note: there may be some bug in `dcnn` modeling, if you want to pretraining.
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
        'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
        'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
        'electra': (ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer),
    }


class DATASET_TYPE:
    # for data reader for each task
    DATA_READER = {
        'sst2': BinarySentimentAnalysisDataReader,
        'imdb': BinarySentimentAnalysisDataReader,
        'agnews': AgnewsReader,
        'mr': BinarySentimentAnalysisDataReader,
        'snli': SnliDataReader
    }

    @staticmethod
    def get_loss_function(dataset: str, reduction='none'):
        if DataReader.OUTPUT_MODE == 'regression':
            return MSELoss(reduction=reduction)
        else:
            return CrossEntropyLoss(reduction=reduction)

    @staticmethod
    def get_evaluation_metric(dataset: str, compare_key: str = '-loss'):
        return ClassificationMetric(compare_key)

