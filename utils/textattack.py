import random
import collections
import torch.nn as nn
from typing import List, Tuple, Union
from transformers import PreTrainedTokenizer
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
                                       BERTAttackLi2020WithoutSentenceEncoder,
                                       TextFoolerJin2019,
                                       TextFoolerJin2019WithoutSentenceEncoder,
                                       HotFlipEbrahimi2017)
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import TextAttackDataset, HuggingFaceDataset
from args import ClassifierArgs

class CustomTextAttackDataset(TextAttackDataset):
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
        instances: List[List[Union[str, int]]],
        label_names: List[str],
        label_map=None,
        output_scale_factor=None,
        dataset_columns=None,
        shuffle=False,
    ):
        assert instances is not None or len(instances) == 0
        self._name = name
        self._i = 0
        self.label_map = None
        self.output_scale_factor = None

        if len(instances[0]) == 2:
            self.input_columns, self.output_column = ("text", ), "label"
            self.examples = [{"text":instance[0], "label": instance[1]} for instance in instances]
        else:
            self.input_columns, self.output_column = ("premise", "hypothesis"), "label"
            self.examples = [{"premise":instance[0], "hypothesis": instance[1], "label": instance[2]} for instance in instances]
        
        self.label_names = label_names

        if shuffle:
            random.shuffle(self.examples)

    @classmethod
    def from_instances(cls, name: str, instances: List[List[Union[str, int]]], label_names: List[str]) -> "CustomTextAttackDataset":
        return cls(name, instances, label_names)
 


def build_english_attacker(args: ClassifierArgs, model: HuggingFaceModelWrapper) -> AttackRecipe:
    if args.mode == 'augmentation':
        attacker = PWWSRen2019.build(model)
    else:
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
            if args.remove_attack_constrainst:
                attacker = TextFoolerJin2019WithoutSentenceEncoder.build(model)
            else:
                attacker = TextFoolerJin2019.build(model)
        elif args.attack_method == 'bae':
            if args.remove_attack_constrainst:
                attacker = BERTAttackLi2020WithoutSentenceEncoder.build(model)
            else:
                attacker = BERTAttackLi2020.build(model)
        else:
            attacker = TextBuggerLi2018.build(model)
    return attacker
