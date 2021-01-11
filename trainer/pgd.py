import torch
import torch.nn as nn
from typing import Tuple
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from args import ClassifierArgs
from .gradient import EmbeddingLevelGradientTrainer

class PGDTrainer(EmbeddingLevelGradientTrainer):
    def __init__(self,
                 data_loader: DataLoader,
                 model: nn.Module,
                 loss_function: _Loss,
                 optimizer: Optimizer,
                 lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None):
        EmbeddingLevelGradientTrainer.__init__(self, data_loader, model, loss_function, optimizer, lr_scheduler, writer)

    def train(self, args: ClassifierArgs, batch: Tuple) -> float:
        assert isinstance(batch[0], torch.Tensor)
        batch = tuple(t.cuda() for t in batch)
        golds = batch[3]

        # for PGD-K, clean batch is not used when training
        adv_batch = self.get_adversarial_examples(args, batch)

        self.model.zero_grad()
        self.optimizer.zero_grad()

        # (0) forward
        logits = self.forward(args, adv_batch)[0]
        # (1) backward
        losses = self.loss_function(logits, golds.view(-1))
        loss = torch.mean(losses)
        loss.backward()
        # (2) update
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        return loss.item()