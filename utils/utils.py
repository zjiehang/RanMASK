import torch
from tqdm import tqdm
from nltk.corpus import words
from pypinyin import lazy_pinyin
from typing import Tuple, List, Dict, Set
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizer


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    if len(batch[0]) == 4:
        all_input_ids, all_attention_mask, all_token_type_ids, all_lens = map(torch.stack, zip(*batch))
    else:
        all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))

    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    if len(batch[0]) == 4:
        return all_input_ids, all_attention_mask, all_token_type_ids
    else:
        return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


def xlnet_collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    if len(batch[0]) == 4:
        all_input_ids, all_attention_mask, all_token_type_ids, all_lens = map(torch.stack, zip(*batch))
    else:
        all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, -max_len:]
    all_attention_mask = all_attention_mask[:, -max_len:]
    all_token_type_ids = all_token_type_ids[:, -max_len:]
    if len(batch[0]) == 4:
        return all_input_ids, all_attention_mask, all_token_type_ids
    else:
        return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


def convert_dataset_to_batch(dataset: TensorDataset, model_type: str):
    batch = tuple(zip(*dataset.tensors))
    if model_type in ['xlnet']:
        return xlnet_collate_fn(batch)
    else:
        return collate_fn(batch)

# batch
def convert_batch_to_bert_input_dict(batch:Tuple[torch.Tensor]=None, model_type:str=None):
    '''
    :param model_type: model type for example, 'bert'
    :param batch: tuple, contains 3 element, batch[0]: embedding
            batch[1]:attention_mask, batch[2]: token_type_ids
    :return:
    '''
    def prepare_token_type_ids(type_ids: torch.Tensor, model_type: str)-> torch.Tensor:
        if model_type in ['bert', 'xlnet', 'albert','roberta']:
            return type_ids
        else:
            return None

    inputs = {}
    if len(batch[0].shape) == 3:
        inputs['inputs_embeds'] = batch[0]
    else:
        inputs['input_ids'] = batch[0]
    inputs['attention_mask'] = batch[1]
    # for distilbert and dcnn, token_type_ids is unnecessary
    if model_type != 'distilbert' and model_type != 'dcnn':
        inputs['token_type_ids'] = prepare_token_type_ids(batch[2], model_type)
    return inputs


def convert_token_list_to_str(tokens: List[str], language: str = 'english'):
    return ' '.join(tokens)


def tokenizer_two_string_to_one(text_a: str, text_b: str = None, language: str = 'english', sep_token : str = "<SEP>"):
    if text_b is None:
        return text_a
    else:
        return '{} {} {}'.format(text_a, sep_token, text_b)


def convert_ids_to_tokens(batch: torch.Tensor, tokenizer: PreTrainedTokenizer):
    pad_ids = batch != tokenizer.pad_token_id
    if len(pad_ids.shape) == 1:
        return [''.join(tokenizer.convert_ids_to_tokens(batch[pad_ids]))]
    else:
        token_list = []
        for token_ids, pad_ids in zip(batch, pad_ids):
            token_list.append(''.join(tokenizer.convert_ids_to_tokens(token_ids[pad_ids])))
        return token_list


def build_confusion_set(path: str) -> Dict:
    confusion_set = {}
    with open(path, 'r', encoding='utf8') as file:
        for line in file.readlines():
            line_split = line.strip().split(':')
            confusion_set[line_split[0]] = line_split[1].split('\t')
            confusion_set[line_split[0]].insert(0, lazy_pinyin(line_split[0])[0])
    return confusion_set

# return type: set, to improve the search complexity. python set: O(1)
def build_forbidden_mask_words(file_path: str) -> Set[str]:
    sentiment_words_set = set()
    with open(file_path, 'r', encoding='utf8') as file:
        for line in file.readlines():
            sentiment_words_set.add(line.strip())
    return sentiment_words_set

