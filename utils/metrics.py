# -*- coding: utf-8 -*-
import torch
import scipy
import numpy as np
from abc import ABC, abstractmethod
from functools import lru_cache
from overrides import overrides
from sklearn.metrics import f1_score, classification_report
from typing import Dict

BASE_FOR_CLASSIFICATION = ['loss', 'accuracy', 'f1']

class Metric(ABC):
    def __init__(self, compare_key='-loss'):
        compare_key = compare_key.lower()
        if not compare_key.startswith('-') and compare_key[0].isalnum():
            compare_key = "+".format(compare_key)
        self.compare_key = compare_key

    def __str__(self):
        return ', '.join(['{}: {:.4f}'.format(key, value) for (key, value) in self.get_metric().items()])

    @abstractmethod
    def __call__(self, ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_metric(self, reset: bool = False) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    def __gt__(self, other: "Metric"):
        is_large = self.compare_key.startswith('+')
        key = self.compare_key[1:]
        assert key in self.get_metric()

        if is_large:
            return self.get_metric()[key] > other.get_metric()[key]
        else:
            return self.get_metric()[key] < other.get_metric()[key]

    def __ge__(self, other: "Metric"):
        is_large = self.compare_key.startswith('+')
        key = self.compare_key[1:]
        assert key in self.get_metric()

        if is_large:
            return self.get_metric()[key] >= other.get_metric()[key]
        else:
            return self.get_metric()[key] <= other.get_metric()[key]


class ClassificationMetric(Metric):
    def __init__(self, compare_key='-loss'):
        super().__init__(compare_key)
        self._all_losses = torch.FloatTensor()
        self._all_predictions = torch.LongTensor()
        self._all_gold_labels = torch.LongTensor()


    @overrides
    def __call__(self,
                 losses: torch.Tensor,
                 logits: torch.FloatTensor,
                 gold_labels: torch.LongTensor
                 ) -> None:
        self._all_losses = torch.cat([self._all_losses, losses.to(self._all_losses.device)], dim=0)
        predictions = logits.argmax(-1).to(self._all_predictions.device)
        self._all_predictions = torch.cat([self._all_predictions, predictions], dim=0)
        self._all_gold_labels = torch.cat([self._all_gold_labels, gold_labels.to(self._all_gold_labels.device)],dim=0)

    @overrides
    def get_metric(self, reset: bool = False) -> Dict:
        loss = torch.mean(self._all_losses).item()
        total_num = self._all_gold_labels.shape[0]
        accuracy = torch.sum(self._all_gold_labels == self._all_predictions).item() / total_num
        f1 = f1_score(y_true=self._all_gold_labels.numpy(), y_pred=self._all_predictions.numpy(), average='macro')
        result = {'loss': loss,
                  'accuracy': accuracy,
                  'f1': f1}
        if reset:
            self.reset()
        return result

    @overrides
    def reset(self) -> None:
        self._all_losses = torch.FloatTensor()
        self._all_predictions = torch.LongTensor()
        self._all_gold_labels = torch.LongTensor()


class RandomSmoothAccuracyMetrics(Metric):
    def __init__(self, compare_key='+accuracy'):
        super().__init__(compare_key)
        self.ABSTAIN_FLAG = -1
        self._all_numbers = 0
        self._abstain_numbers = 0
        self._correct_numbers = 0

    @overrides
    def __call__(self,
                 scores: np.ndarray, 
                 target: int,
                 alpha: float,
                 ) -> None:
        label_num = scores.shape[1]
        preds = np.argmax(scores, axis=-1)
        votes = np.bincount(preds, minlength=label_num)
        top_two_idx = np.argsort(votes)[-2:][::-1]
        top_one_count, top_two_count = votes[top_two_idx[0]], votes[top_two_idx[1]]
        tests = scipy.stats.binom_test(top_one_count,top_one_count+top_two_count, .5)
        if tests <= alpha:
            pred = top_two_idx[0]
            if target == pred:
                self._correct_numbers += 1
        else:
            self._abstain_numbers += 1
        self._all_numbers += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Dict:
        result = {'accuracy': self._correct_numbers / self._all_numbers,
                  'abstain': self._abstain_numbers / self._all_numbers}
        if reset:
            self.reset()
        return result

    @overrides
    def reset(self) -> None:
        self._all_numbers = 0
        self._abstain_numbers = 0
        self._correct_numbers = 0


