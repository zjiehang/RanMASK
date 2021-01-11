import os
import torch
from typing import List
from tqdm import tqdm
from data.instance import InputFeature, InputInstance
from data.reader import DataReader
from transformers import PreTrainedTokenizer
from torch.utils.data.dataset import TensorDataset, Dataset
from data.dataset import ListDataset
'''
data processor: reading data, tokenizing data (if needed), converting data to TensorDataset
'''

class DataProcessor:
    def __init__(self,
                 data_reader: DataReader,
                 tokenizer: PreTrainedTokenizer,
                 model_type: str,
                 max_seq_length: int = 256):
        self.data_reader = data_reader
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.max_seq_length = max_seq_length

    def _convert_instances_to_features(self,
                                      instances: List[InputInstance],
                                      mask_padding_with_zero: bool = True,
                                      use_tqdm: bool = True) -> List[InputFeature]:
        '''
        Loads a data file into a list of ``InputFeature``
        :param instances: List of ``InputInstances`` or ``tf.data.Dataset`` containing the instances.
        :param mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
        :return:
            features: List of ``InputFeature``
        '''

        # label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        label_list = self.data_reader.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}

        # String indicating the output mode. Either ``regression`` or ``classification``
        output_mode = self.data_reader.OUTPUT_MODE

        # pad_on_left: If set to ``True``, the instances will be padded on the left rather than on the right (default)
        # pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        pad_on_left = bool(self.model_type in ["xlnet"])
        pad_token_segment_id = 4 if self.model_type in ["xlnet"] else 0

        # pad_token_id
        pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]

        features = []
        if use_tqdm:
            iterator = tqdm(instances)
        else:
            iterator = instances
        for instance in iterator:
            inputs = self.tokenizer.encode_plus(
                instance.text_a,
                instance.text_b,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_seq_length
            )
            input_ids = inputs["input_ids"]
            if "token_type_ids" in inputs:
                token_type_ids = inputs["token_type_ids"]
            else:
                token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(input_ids[1:-1])

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            input_len = len(input_ids)
            # Zero-pad up to the sequence length.
            padding_length = self.max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == self.max_seq_length, "Error with input length {} vs {}".format(len(input_ids), self.max_seq_length)
            assert len(attention_mask) == self.max_seq_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                                self.max_seq_length)
            assert len(token_type_ids) == self.max_seq_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                                self.max_seq_length)

            if instance.label is not None:
                if output_mode == "classification":
                    label = label_map[instance.label]
                elif output_mode == "regression":
                    label = float(instance.label)
                else:
                    raise KeyError(output_mode)
            else:
                label = None

            features.append(
                InputFeature(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label,
                              input_len=input_len))

        return features

    def _convert_features_to_tensor_dataset(self, features: List[InputFeature]) -> TensorDataset:
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)

        none_labels = any([True if f.label is None else False for f in features])
        if none_labels:
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens)
        else:
            if self.data_reader.OUTPUT_MODE == "classification":
                all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            elif self.data_reader.OUTPUT_MODE == "regression":
                all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
            else:
                all_labels = None
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)
        return dataset

    def convert_instances_to_dataset(self, instances: List[InputInstance], tokenizer: bool = True, use_tqdm: bool=True) -> Dataset:
        '''
        convert list of input instances to dataset, and then can be used in dataloader
        :param instances: list of input instances
        :param tokenizer: whether to tokenizer instances, for some training type, it is better to use instance as input directly,
                           for example, when you want to augment the instances
        :param use_tqdm: whether to use tqdm when convert instances features
        :return:
        '''
        if tokenizer:
            features = self._convert_instances_to_features(instances, use_tqdm=use_tqdm)
            dataset = self._convert_features_to_tensor_dataset(features)
        else:
            dataset = ListDataset(instances)
        return dataset

    def read_from_file(self, path: str, data_type: str, tokenizer: bool = True) -> Dataset:
        '''
        :param path:  data dir
        :param data_type: in 'train' 'dev' 'test '
        :param data_file_name: file_name to load, if None, the same as data_type
        :param tokenizer: whether to tokenizer the instances
        :return:
        '''
        assert data_type in ['train', 'dev', 'test']
        assert os.path.exists(path) and os.path.isdir(path)

        instances = self.data_reader.get_instances(path, data_type)

        dataset = self.convert_instances_to_dataset(instances, tokenizer)
        return dataset

    def read_from_content_batch(self,
                                text_a_list: List[str],
                                text_b_list: List[str] = None,
                                label_list: List[str] = None,
                                tokenizer: bool = True) -> Dataset:
        '''
        :param text_a_list:
        :param text_b_list:
        :param label_list:
        :param tokenizer:
        :return:
        '''
        if text_b_list is None:
            text_b_list = [None] * len(text_a_list)
        if label_list is None:
            label_list = [None] * len(text_a_list)
        instances = []
        for index, (text_a, text_b, label) in enumerate(zip(text_a_list, text_b_list, label_list)):
            instances.append(InputInstance(guid='predict{}'.format(index),
                                         text_a=text_a,
                                         text_b=text_b,
                                         label=label))
        dataset = self.convert_instances_to_dataset(instances, tokenizer=tokenizer, use_tqdm=False)
        return dataset

    def read_from_content(self,
                          text_a: str,
                          text_b: str = None,
                          label: str = None,
                          tokenizer: bool = True) -> Dataset:
        instance = InputInstance(guid='predict',
                               text_a=text_a,
                               text_b=text_b,
                               label=label)
        dataset = self.convert_instances_to_dataset([instance], tokenizer=tokenizer, use_tqdm=False)
        return dataset