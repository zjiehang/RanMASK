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
from args import ClassifierArgs
from data.instance import InputInstance
from data.processor import DataProcessor
from utils.utils import convert_dataset_to_batch,collate_fn, xlnet_collate_fn, build_forbidden_mask_words
from utils.safer import WordSubstitude
from overrides import overrides
from .base import BaseTrainer

'''
SAFER trainer, from paper: https://arxiv.org/pdf/2005.14424.pdf
SAFER: A Structure-free Approach for Certified Robustness to Adversarial Word Substitutions
'''
class SAFERTrainer(BaseTrainer):
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
        self.augmentor = WordSubstitude(args.safer_perturbation_set)
    

    def train(self, args: ClassifierArgs, batch: Tuple) -> float:
        assert isinstance(batch[0], InputInstance)
        perturb_instances = self.perturb_batch(batch)
        train_batch =  self.data_processor.convert_instances_to_dataset(perturb_instances, use_tqdm=False)
        train_batch = convert_dataset_to_batch(train_batch, args.model_type)
        return super().train(args, train_batch)

    def perturb_batch(self, instances: List[InputInstance]) -> List[InputInstance]:
        result_instances = []
        for instance in instances:
            perturb_sentences = self.augmentor.get_perturbed_batch(instance.perturbable_sentence().lower())
            tmp_instances = []
            for sentence in perturb_sentences:
                tmp_instances.append(InputInstance.from_instance_and_perturb_sentence(instance, sentence))
            result_instances.extend(tmp_instances)
        return result_instances