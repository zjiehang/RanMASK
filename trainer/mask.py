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
from utils.utils import convert_dataset_to_batch,collate_fn, xlnet_collate_fn, build_forbidden_mask_words
from utils.mask import mask_instance, mask_forbidden_index
from utils.config import DATASET_TYPE
from overrides import overrides
from .base import BaseTrainer

class MaskTrainer(BaseTrainer):
    def __init__(self,
                 args: ClassifierArgs, 
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
        # self.task = args.task
        self.forbidden_words = None
        if args.keep_sentiment_word:
            self.forbidden_words = build_forbidden_mask_words(args.sentiment_path)

        # for incremental trick
        # mask rate = initial + Δ * self.global_step
        # where Δ is the incremental mask rate for each global step
        self.initial_mask_rate = args.initial_mask_rate
        self.incremental_trick = args.incremental_trick
        time_for_epoch = self.training_times_in_epoch(len(data_loader.dataset), args.batch_size)
        self.delta = (args.sparse_mask_rate - self.initial_mask_rate) / (time_for_epoch * (args.epochs - round(args.epochs * 0.4)))


    def training_times_in_epoch(self, data_len: int, batch_size: int) -> int:
        if data_len % batch_size == 0:
            return data_len // batch_size
        else:
            return data_len // batch_size + 1
        
    def cal_sparse_mask_rate(self, sparse_mask_rate):
        if self.incremental_trick:
            tmp_mask_rate = self.initial_mask_rate + self.global_step * self.delta
            if tmp_mask_rate < sparse_mask_rate:
                return tmp_mask_rate
            else:
                return sparse_mask_rate
        else:
            return sparse_mask_rate
        
    def train(self, args: ClassifierArgs, batch: Tuple) -> float:
        assert isinstance(batch[0], InputInstance)
        mask_rate = self.cal_sparse_mask_rate(args.sparse_mask_rate)
        mask_instances = self.mask_batch_instances(batch, mask_rate, self.mask_token)
        train_batch =  self.data_processor.convert_instances_to_dataset(mask_instances, use_tqdm=False)
        train_batch = convert_dataset_to_batch(train_batch, args.model_type)
        return super().train(args, train_batch)

    def mask_batch_instances(self, instances: List[InputInstance], rate: float, token: str, nums: int = 1) -> List[InputInstance]: 
        batch_instances = []
        for instance in instances:
            forbidden_index = None
            if self.forbidden_words is not None:
                forbidden_index = mask_forbidden_index(instance.perturbable_sentence(), self.forbidden_words)
            batch_instances.extend(mask_instance(instance, rate, token, nums, forbidden_indexes=forbidden_index))
        return batch_instances