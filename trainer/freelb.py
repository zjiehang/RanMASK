import torch
import torch.nn as nn
from typing import Tuple
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from args import ClassifierArgs
from .gradient import EmbeddingLevelGradientTrainer

class FreeLBTrainer(EmbeddingLevelGradientTrainer):
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
        word_embedding_layer = self.model.get_input_embeddings()

        # init input_ids and mask
        batch_in_token_ids = batch[0]
        attention_mask = batch[1]
        golds = batch[3]
        embedding_init = word_embedding_layer(batch_in_token_ids)

        delta = EmbeddingLevelGradientTrainer.delta_initial(args, embedding_init, attention_mask)

        total_loss = 0.0
        for astep in range(args.adv_steps):
            # (0) forward
            delta.requires_grad_()
            batch = (delta + embedding_init, batch[1], batch[2])
            logits = self.forward(args, batch)[0]

            # (1) backward
            losses = self.loss_function(logits, golds.view(-1))
            loss = torch.mean(losses)
            loss = loss / args.adv_steps
            total_loss += loss.item()
            loss.backward()

            if astep == args.adv_steps - 1:
                break

            # (2) get gradient on delta
            delta_grad = delta.grad.clone().detach()

            # (3) update and clip
            delta = EmbeddingLevelGradientTrainer.delta_update(args, embedding_init, delta, delta_grad)
            embedding_init = word_embedding_layer(batch_in_token_ids)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        return total_loss