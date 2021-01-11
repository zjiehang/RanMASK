import numpy as np
from typing import List
from data.instance import InputInstance

def mask_batch_instances(instances: List[InputInstance], rate: float, token: str, nums: int = 1) -> List[InputInstance]: 
    batch_instances = []
    for instance in instances:
        batch_instances.extend(mask_instance(instance, rate, token, nums))
    return batch_instances

def mask_instance(instance: InputInstance, rate: float, token: str, nums: int = 1):
    sentence = instance.perturbable_sentence()
    
    # str --> List[str]
    sentence_in_list = sentence.split()
    length = len(sentence_in_list)
    mask_numbers = round(length * rate)
    mask_indexes = [np.random.choice(list(range(length)), mask_numbers, replace=False) for _ in range(nums)]
    tmp_instances = []
    for indexes in mask_indexes:
        tmp_sentence = mask_sentence(sentence_in_list, indexes, token)
        tmp_instance = InputInstance.from_instance_and_perturb_sentence(instance, tmp_sentence)
        tmp_instances.append(tmp_instance)
    return tmp_instances

def mask_sentence(sentence: List[str], indexes: np.ndarray, token: str) -> str:
    tmp_sentence = sentence.copy()
    for index in indexes:
        tmp_sentence[index] = token
    return ' '.join(tmp_sentence)
