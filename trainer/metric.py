import torch
import torch.nn as nn
from typing import Tuple
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizer
from args import ClassifierArgs
from utils.hook import EmbeddingHook
from torch.nn import functional as F
from .gradient import GradientTrainer, EmbeddingLevelGradientTrainer
from .hotflip import HotflipTrainer

class MetricTrainer(GradientTrainer):
    def __init__(self,
                 data_loader: DataLoader,
                 model: nn.Module,
                 loss_function: _Loss,
                 optimizer: Optimizer,
                 lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None):
        GradientTrainer.__init__(self, data_loader, model, loss_function, optimizer, lr_scheduler, writer)

    def train(self, args: ClassifierArgs, batch: Tuple) -> float:
        assert isinstance(batch[0], torch.Tensor)
        batch = tuple(t.cuda() for t in batch)
        golds = batch[3]

        # get adversarial examples
        adv_batch = self.get_adversarial_examples(args, batch)

        # clean loss
        logits, hidden_states = self.forward(args, batch)
        losses = self.loss_function(logits, golds.view(-1))
        loss = torch.mean(losses)
        loss.backward(retain_graph=True)
        clean_rep = F.normalize(hidden_states[-1][:, 0], dim=-1, p=2)

        adv_logits, adv_hidden_states = self.forward(args, adv_batch)
        adv_losses = self.loss_function(adv_logits, golds.view(-1))
        adv_loss = torch.mean(adv_losses)
        adv_loss.backward(retain_graph=True)
        adv_rep = F.normalize(adv_hidden_states[-1][:, 0], dim=-1, p=2)

        # metric loss (triplet loss)
        # default is cosine-similarity distance
        # triplet loss = ( distance(p, a) )
        distances = 1.0 - torch.matmul(adv_rep, clean_rep.transpose(0, 1))
        # label masking
        l = golds.unsqueeze(-1).repeat(1, golds.size(0)).transpose(0, 1)
        mask = [l[i] == golds[i] for i in range(golds.size(0))]
        mask = torch.stack(mask, dim=0)
        positive = distances.diag()
        negative = distances.masked_fill(mask, 20181314.).min(dim=-1).values
        metric_loss = torch.mean(F.relu(positive - negative + args.metric_learning_margin))
        metric_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        return loss.item() + adv_loss.item() + metric_loss.item()


class EmbeddingLevelMetricTrainer(MetricTrainer, EmbeddingLevelGradientTrainer):
    def __init__(self,
                 data_loader: DataLoader,
                 model: nn.Module,
                 loss_function: _Loss,
                 optimizer: Optimizer,
                 lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None):
        MetricTrainer.__init__(self, data_loader, model, loss_function, optimizer, lr_scheduler, writer)
        EmbeddingLevelGradientTrainer.__init__(self, data_loader, model, loss_function, optimizer, lr_scheduler, writer)


class TokenLevelMetricTrainer(MetricTrainer, HotflipTrainer):
    def __init__(self,
                 args: ClassifierArgs,
                 tokenizer: PreTrainedTokenizer,
                 data_loader: DataLoader,
                 model: nn.Module,
                 loss_function: _Loss,
                 optimizer: Optimizer,
                 lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None):
        MetricTrainer.__init__(self, data_loader, model, loss_function, optimizer, lr_scheduler, writer)
        HotflipTrainer.__init__(self, args, tokenizer, data_loader, model, loss_function, optimizer, lr_scheduler, writer)
