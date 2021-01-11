import torch
import torch.nn as nn
from typing import Tuple
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from data.instance import InputInstance
from data.processor import DataProcessor
from args import ClassifierArgs
from utils.loss import ContrastiveLearningLoss, UnsupervisedCircleLoss
from utils.utils import convert_dataset_to_batch
from .base import BaseTrainer


class RepresentationLearningTrainer(BaseTrainer):
    def __init__(self,
                 augmentor,
                 data_processor: DataProcessor,
                 data_loader: DataLoader,
                 model: nn.Module,
                 loss_function: _Loss,
                 optimizer: Optimizer,
                 lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None,
                 type: str = 'con'):
        BaseTrainer.__init__(self, data_loader, model, loss_function, optimizer, lr_scheduler, writer)

        self.augmentor = augmentor
        self.data_processor = data_processor
        self.rep_loss = UnsupervisedCircleLoss() if type == 'circle' else ContrastiveLearningLoss()

    def train(self, args: ClassifierArgs, batch: Tuple) -> float:
        assert isinstance(batch[0], InputInstance)
        aug_batch = []
        for instance in batch:
            aug_batch.extend([instance])
            tmp_batch = self.augmentor.augment(instance)
            aug_batch.extend(tmp_batch)
        aug_dataset = self.data_processor.convert_instances_to_dataset(aug_batch, use_tqdm=False)
        aug_batch = convert_dataset_to_batch(aug_dataset, args.model_type)
        aug_batch = tuple(t.cuda() for t in aug_batch)
        logits, hidden_states = self.forward(args, aug_batch)

        logits = logits[::(args.augmentation_numbers + 1)]
        golds = aug_batch[3][::(args.augmentation_numbers + 1)]
        cls_losses = self.loss_function(logits, golds.view(-1))
        cls_loss = torch.mean(cls_losses)
        features = torch.stack(torch.split(hidden_states, args.augmentation_numbers + 1))
        unsup_loss = args.unrep_loss * self.rep_loss(features)
        sup_loss = args.rep_loss * self.rep_loss(features, golds)
        loss = cls_loss + unsup_loss + sup_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        return loss.item()