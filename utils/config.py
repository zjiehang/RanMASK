from torch.nn import MSELoss, CrossEntropyLoss
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer
from transformers import ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer
from transformers import AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from utils.metrics import ClassificationMetric
from data.reader import *

class CONFIG:
    ABSTAIN_FLAG = -1

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
    # TEXT_CLASSIFIER_TASK = 'classifier'
    # SENTIMENT_ANALYSIS_TASK = 'sentiment'
    # TEXT_ENTAILMENT_TASK = 'entailment'

    # for data reader for each task
    DATA_READER = {
        'sst2': BinarySentimentAnalysisDataReader,
        'imdb': BinarySentimentAnalysisDataReader,
        'agnews': AgnewsReader,
        'mr': BinarySentimentAnalysisDataReader,
        'snli': SnliDataReader
    }

    # TASK = {
    #     'sst2': SENTIMENT_ANALYSIS_TASK,
    #     'imdb': SENTIMENT_ANALYSIS_TASK,
    #     'agnews': TEXT_CLASSIFIER_TASK,
    #     'mr': SENTIMENT_ANALYSIS_TASK,
    #     'snli': TEXT_ENTAILMENT_TASK
    # }

    # @staticmethod
    # def get_task(dataset):
    #     assert dataset in DATASET_TYPE.TASK.keys()
    #     return DATASET_TYPE.TASK[dataset]

    @staticmethod
    def get_loss_function(dataset: str, reduction='none'):
        if DataReader.OUTPUT_MODE == 'regression':
            return MSELoss(reduction=reduction)
        else:
            return CrossEntropyLoss(reduction=reduction)

    @staticmethod
    def get_evaluation_metric(dataset: str, compare_key: str = '-loss'):
        return ClassificationMetric(compare_key)

