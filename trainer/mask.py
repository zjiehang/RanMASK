import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
from transformers import PreTrainedTokenizer
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SequentialSampler
from torchnlp.samplers.bucket_batch_sampler import BucketBatchSampler
from args import ClassifierArgs
from data.instance import InputInstance
from data.processor import DataProcessor
from utils.loss import ContrastiveLearningLoss, UnsupervisedCircleLoss
from utils.utils import convert_dataset_to_batch,collate_fn, xlnet_collate_fn
from utils.mask import mask_batch_instances
from .base import BaseTrainer

class MaskTrainer(BaseTrainer):
    def __init__(self,
                 data_processor: DataProcessor,
                 data_loader: DataLoader,
                 model: nn.Module,
                 loss_function: _Loss,
                 optimizer: Optimizer,
                 lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None):
        BaseTrainer.__init__(self, data_loader, model, loss_function, optimizer, lr_scheduler, writer)

        self.data_processor = data_processor
        self.mask_token = self.data_processor.tokenizer.mask_token


    # def mask_batch(self, batchs: List[InputInstance], mask_rate: float) -> List[InputInstance]:
    #     mask_instances = []
    #     for instance in batchs:
    #         if instance.text_b is None:
    #             sentence = instance.text_a
    #         else:
    #             sentence = instance.text_b
            
    #         mask_sentence = self.mask_sentence(sentence, mask_rate)

    #         if instance.text_b is None:
    #             tmp_instance = InputInstance(instance.guid, text_a=mask_sentence, label=instance.label)
    #         else:
    #             tmp_instance = InputInstance(instance.guid, text_a=instance.text_a, text_b=mask_sentence, label=instance.label)
            
    #         mask_instances.append(tmp_instance)

    #     return mask_instances

    # def mask_sentence(self, sentence: str, mask_rate: float) -> str:
    #     sentence_split = sentence.split()
    #     length = len(sentence_split)
    #     mask_nums = round(length * mask_rate)
    #     mask_token_ids = np.random.choice(list(range(length)), size=(mask_nums,),replace=False)
    #     for token_ids in mask_token_ids:
    #         sentence_split[token_ids] = self.mask_token
    #     return ' '.join(sentence_split)

    def train(self, args: ClassifierArgs, batch: Tuple) -> float:
        assert isinstance(batch[0], InputInstance)
        mask_instances = mask_batch_instances(batch, args.sparse_mask_rate, self.mask_token)
        train_batch =  self.data_processor.convert_instances_to_dataset(mask_instances, use_tqdm=False)
        train_batch = convert_dataset_to_batch(train_batch, args.model_type)
        return super().train(args, train_batch)
        