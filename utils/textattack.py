import random
import collections
import torch.nn as nn
from typing import List, Tuple, Union, Dict
from transformers import PreTrainedTokenizer
from data.instance import InputInstance
from textattack.attack_recipes import AttackRecipe
from textattack.attack_recipes import (PWWSRen2019,
                                       GeneticAlgorithmAlzantot2018,
                                       GeneticAlgorithmAlzantot2018WithoutLM,
                                       FasterGeneticAlgorithmJia2019,
                                       FasterGeneticAlgorithmJia2019WithoutLM,
                                       DeepWordBugGao2018,
                                       PSOZang2020,
                                       TextBuggerLi2018,
                                       BERTAttackLi2020,
                                       TextFoolerJin2019,
                                       HotFlipEbrahimi2017)
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import TextAttackDataset, HuggingFaceDataset
from textattack.attack_results.attack_result import AttackResult
from textattack.attack_results import SuccessfulAttackResult, SkippedAttackResult, FailedAttackResult
from args import ClassifierArgs

class SimplifidResult:
    def __init__(self,):
        self._succeed = 0
        self._fail = 0
        self._skipped = 0

    def __str__(self):
        return ', '.join(['{}: {:.2f}%'.format(key, value) for (key, value) in self.get_metric().items()])

    def __call__(self, result: AttackResult) -> None:
        assert isinstance(result, AttackResult)
        if isinstance(result, SuccessfulAttackResult):
            self._succeed += 1
        elif isinstance(result, FailedAttackResult):
            self._fail += 1
        elif isinstance(result, SkippedAttackResult):
            self._skipped += 1

    def get_metric(self, reset: bool = False) -> Dict:
        all_numbers = self._succeed + self._fail + self._skipped
        correct_numbers = self._succeed + self._fail

        if correct_numbers == 0:
            success_rate = 0.0
        else:
            success_rate = self._succeed / correct_numbers
        
        if all_numbers == 0:
            clean_accuracy = 0.0
            robust_accuracy = 0.0
        else:
            clean_accuracy = correct_numbers / all_numbers
            robust_accuracy = self._fail / all_numbers
        
        if reset:
            self.reset()

        return {"Accu(cln)" : clean_accuracy * 100, 
                "Accu(rob)" : robust_accuracy * 100,
                "Succ": success_rate * 100}

    def reset(self) -> None:
        self._succeed += 1
        self._fail += 1
        self._skipped += 1        


class CustomTextAttackDataset(HuggingFaceDataset):
    """Loads a dataset from HuggingFace ``datasets`` and prepares it as a
    TextAttack dataset.

    - name: the dataset name
    - subset: the subset of the main dataset. Dataset will be loaded as ``datasets.load_dataset(name, subset)``.
    - label_map: Mapping if output labels should be re-mapped. Useful
      if model was trained with a different label arrangement than
      provided in the ``datasets`` version of the dataset.
    - output_scale_factor (float): Factor to divide ground-truth outputs by.
        Generally, TextAttack goal functions require model outputs
        between 0 and 1. Some datasets test the model's correlation
        with ground-truth output, instead of its accuracy, so these
        outputs may be scaled arbitrarily.
    - shuffle (bool): Whether to shuffle the dataset on load.
    """
    def __init__(
        self,
        name,
        instances: List[InputInstance],
        label_names: List[str] = None,
        label_map: Dict[str, int] = None,
        output_scale_factor=None,
        dataset_columns=None,
        shuffle=False,
    ):
        assert instances is not None or len(instances) == 0
        self._name = name
        self._i = 0
        self.label_map = label_map
        self.output_scale_factor = output_scale_factor
        self.label_names = label_names

        if instances[0].is_nli():
            self.input_columns, self.output_column = ("premise", "hypothesis"), "label"
            self.examples = [{"premise":instance.text_a, "hypothesis": instance.text_b, "label": instance.label} for instance in instances]
        else:
            self.input_columns, self.output_column = ("text", ), "label"
            self.examples = [{"text":instance.text_a, "label": instance.label} for instance in instances]
        
        if shuffle:
            random.shuffle(self.examples)

    @classmethod
    def from_instances(cls, name: str, instances: List[InputInstance], labels: List[str]) -> "CustomTextAttackDataset":
        return cls(name, instances, labels, {label: index for index, label in enumerate(labels)})
 

def build_english_attacker(args: ClassifierArgs, model: HuggingFaceModelWrapper) -> AttackRecipe:
    if args.attack_method == 'pwws':
        attacker = PWWSRen2019.build(model)
    elif args.attack_method == 'pso':
        attacker = PSOZang2020.build(model)
    elif args.attack_method == 'ga':
        if args.remove_attack_constrainst:
            attacker = GeneticAlgorithmAlzantot2018WithoutLM.build(model)
        else:
            attacker = GeneticAlgorithmAlzantot2018.build(model)
    elif args.attack_method == 'fga':
        if args.remove_attack_constrainst:
            attacker = FasterGeneticAlgorithmJia2019WithoutLM.build(model)
        else:
            attacker = FasterGeneticAlgorithmJia2019.build(model)
    elif args.attack_method == 'textfooler':
        attacker = TextFoolerJin2019.build(model)
    elif args.attack_method == 'bae':
        attacker = BERTAttackLi2020.build(model)
    elif args.attack_method == 'deepwordbug':
        attacker = DeepWordBugGao2018.build(model)
    elif args.attack_method == 'textbugger':
        attacker = TextBuggerLi2018.build(model)
    else:
        attacker = TextBuggerLi2018.build(model)
    return attacker
