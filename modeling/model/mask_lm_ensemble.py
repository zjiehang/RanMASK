import math
import torch
import numpy as np
from copy import copy
from overrides import overrides
from typing import List, Tuple, Dict, Union
import torch.nn as nn
from transformers import BertForMaskedLM, BertTokenizer, PreTrainedTokenizer, RobertaTokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from args import ClassifierArgs


class MaskLMEnsembleModel(nn.Module):
    def __init__(self, 
                args: ClassifierArgs,
                classifier: nn.Module, 
                tokenizer: PreTrainedTokenizer):
        super().__init__()

        self.detector = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
        self.tokenizer = tokenizer
        self.classifier = classifier

        self.random_mask = args.random_mask
        self.mask_rate = args.mask_rate
        self.mask_ensemble_numbers = args.mask_ensemble_numbers
        self.max_length = args.max_seq_length

    def get_tokenizer_mapping_for_sentence(self, sentence: str) -> Tuple:
        if isinstance(self.tokenizer, RobertaTokenizer):
            sentence_tokens = sentence.split()
            enc_result = [self.tokenizer.encode(sentence_tokens[0], add_special_tokens=False)]
            enc_result.extend([self.tokenizer.encode(x, add_special_tokens=False, add_prefix_space=True) for x in sentence_tokens[1:]])
        else:
            enc_result = [self.tokenizer.encode(x, add_special_tokens=False) for x in sentence.split()]
        desired_output = []
        idx = 1
        for token in enc_result:
            tokenoutput = []
            for _ in token:
                tokenoutput.append(idx)
                idx += 1
            desired_output.append(tokenoutput)
        return (enc_result, desired_output)

    def batch_encode_plus(self, sentences: List[str], cuda: bool = True) -> Dict:
        encodings = self.tokenizer.batch_encode_plus(sentences,
                                                    truncation=True,
                                                    max_length=self.max_length,
                                                    add_special_tokens=True,
                                                    padding=True,
                                                    return_tensors='pt')
        if cuda:
            encodings = {key: value.cuda() for key, value in encodings.items()}
        return encodings

    def get_word_loss(self, indexes: List[int], losses: torch.Tensor) -> float:
        try:
            loss = []
            for index in indexes:
                if index <= self.max_length - 2:
                    loss.append(losses[index].item())
                else:
                    loss.append(0.0)
            return np.mean(loss)
        except:
            return 0.0


    # def get_sentence_word_scores(self, 
    #                             probs: torch.Tensor, 
    #                             sentence_vocab_index: List[List[int]], 
    #                             sentence_mappding_index:List[List[int]]) -> List[torch.Tensor]:
    #     sentence_logits = []
    #     for vocab_idx, mapping_index in zip(sentence_vocab_index, sentence_mappding_index):
    #         sentence_logits.append(self.get_word_scores(probs, vocab_idx, mapping_index))

    #     sentence_logits = torch.softmax(-1.0 * torch.tensor(sentence_logits), dim=0)
    #     return sentence_logits

    # def get_batch_word_scores(self, batch_probs: torch.Tensor, batch_tokenizer_mapping: List[Tuple]) -> List[List[float]]:        
    #     batch_word_scores = []
    #     for probs, tokenizer_mapping in zip(batch_probs, batch_tokenizer_mapping):
    #         batch_word_scores.append(self.get_sentence_word_scores(probs, tokenizer_mapping[0], tokenizer_mapping[1]))
    #     return batch_word_scores

    def get_sentence_probs(self, sentences: List[str]) -> List[List[float]]:
        encodings = self.batch_encode_plus(sentences)
        logits = self.detector(**encodings)[0]
        losses = self.get_lm_loss(encodings["input_ids"], logits)
        batch_mask_probs = []
        for loss, sentence in zip(losses, sentences):
            _, mappings = self.get_tokenizer_mapping_for_sentence(sentence)
            sentence_mask_probs = [self.get_word_loss(mapping, loss) for mapping in mappings]
            
            # sum to one
            sentence_mask_probs = self.sum_to_one(sentence_mask_probs)
            batch_mask_probs.append(sentence_mask_probs)
        return batch_mask_probs

    # div by sum, consider another method if necessary
    def sum_to_one(self, logits: List[float]) -> List[float]:
        return logits / sum(logits)
        

    def mask_sentence_by_indexes(self, sentence: List[str], indexes: List[int], mask_token: str = '[MASK]') -> List[str]:
        tmp_sentence = copy(sentence)
        for index in indexes:
            tmp_sentence[index] = mask_token
        return tmp_sentence

    def get_batch_mask_sentence(self, batch_word_scores: List[List[float]], inputs: List[str]) -> Tuple[List[str],List[List[int]]]:
        mask_sentences = []
        perturbed_indexes = []
        for word_scores, sentence in zip(batch_word_scores, inputs):
            tokenizer_sentence = sentence.split()
            repair_numbers = math.ceil(len(tokenizer_sentence) * self.repair_rate)
            if not self.ensemble:
                topk_index = torch.topk(word_scores, repair_numbers)[1].tolist()
                mask_sentences.append(' '.join(self.mask_sentence_by_indexes(tokenizer_sentence, topk_index, self.corrector_tokenizer.mask_token)))
                perturbed_indexes.append(topk_index)
            else:
                topk_probs, topk_index = torch.topk(word_scores, k=min(self.repair_top_k,len(tokenizer_sentence)))
                topk_probs = (topk_probs / topk_probs.sum()).tolist()
                topk_index = topk_index.tolist()
                for _ in range(self.ensemble_numbers):
                    random_indexes = np.random.choice(topk_index, p=topk_probs,replace=False,size=(min(repair_numbers, self.repair_top_k),)).tolist()
                    mask_sentences.append(' '.join(self.mask_sentence_by_indexes(tokenizer_sentence, random_indexes, self.corrector_tokenizer.mask_token)))
                    perturbed_indexes.append(random_indexes)
        return mask_sentences, perturbed_indexes

    def get_batch_repair_sentence(self, mask_sentence_list: List[str], batch_perturbed_indexes: List[List[int]]) -> List[str]:
        logits = self.batch_predict(self.corrector, self.corrector_tokenizer, mask_sentence_list)
        repair_sentences_list = []
        for logit, mask_sentence, perturbed_index in zip(logits, mask_sentence_list, batch_perturbed_indexes):
            tmp_sentences = mask_sentence.split()
            best_index = torch.argmax(logit, dim=-1).tolist()
            best_tokens = [self.corrector_tokenizer.convert_ids_to_tokens(index) for index in best_index]
            mapping_index = self.get_tokenizer_mapping_for_one_sentence(self.corrector_tokenizer, mask_sentence)[1]
            for index in perturbed_index:
                tmp_sentences[index] = self.repair([best_tokens[i] for i in mapping_index[index]])

            repair_sentences_list.append(' '.join(tmp_sentences))
        return repair_sentences_list

    def get_lm_loss(self, batch_ids: torch.Tensor, batch_pred: torch.Tensor, delete_special_tokens: bool = True) -> torch.Tensor:
        batch_size = batch_ids.shape[0]
        classes = batch_pred.shape[2]
        lm_target = batch_ids.reshape(-1)
        lm_pred = batch_pred.reshape(-1, classes)
        loss = torch.nn.functional.cross_entropy(lm_pred, lm_target, reduction='none')
        loss = loss.reshape(batch_size, -1)
        if delete_special_tokens:
            loss[batch_ids == self.tokenizer.pad_token_id] = 0.0
            loss[batch_ids == self.tokenizer.sep_token_id] = 0.0
            loss[batch_ids == self.tokenizer.cls_token_id] = 0.0
        return loss

    def repeat_dict(self, batch_input: Dict[str, torch.Tensor], ignore_keys: List[str] = None) -> Dict[str, torch.Tensor]:
        result_dict = {}
        for key, value in batch_input.items():
            if ignore_keys is not None:
                if key not in ignore_keys:
                    result_dict[key] = torch.repeat_interleave(value, repeats=self.mask_ensemble_numbers, dim=0)
            else:
                result_dict[key] = torch.repeat_interleave(value, repeats=self.mask_ensemble_numbers, dim=0)
        return result_dict

    # def mask_token_by_random(self, batch_ids: torch.Tensor) -> torch.Tensor:
    #     mask = torch.zeros_like(batch_ids).float().uniform_() > self.mask_rate
    #     mask_batch_ids = batch_ids.masked_fill(mask, value=torch.tensor(self.tokenizer.mask_token_id))
    #     mask_batch_ids[batch_ids == self.tokenizer.cls_token_id] = self.tokenizer.cls_token_id
    #     mask_batch_ids[batch_ids == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
    #     mask_batch_ids[batch_ids == self.tokenizer.sep_token_id] = self.tokenizer.sep_token_id
    #     return mask_batch_ids

    # def mask_token_by_lm(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    #     batch = {key: value.cuda() for key,value in batch.items()}
    #     logits = self.detector(**batch)[0]

    #     losses = self.get_lm_loss(batch["input_ids"], logits)
    #     # losses = losses / torch.sum(losses, dim=-1, keepdim=True)
    #     lengths = torch.sum(batch['attention_mask'], dim=1).tolist()

    #     mask_batch_ids = []
    #     for length, batch_ids, loss in zip(lengths, batch["input_ids"], losses):
    #         indexes = list(range(1, length - 1))
    #         loss = loss[1: length - 1].cpu().numpy()
    #         loss = loss / np.sum(loss)
    #         mask_numbers = math.floor(length * self.mask_rate)
    #         for _ in range(self.mask_ensemble_numbers):
    #             random_mask_ids = np.random.choice(indexes, size=(mask_numbers,), replace=False, p=loss)
    #             tmp_batch_ids = batch_ids.clone()
    #             mask = torch.zeros_like(batch_ids)
    #             mask.scatter_(0, torch.tensor(random_mask_ids).cuda(), 1.0)
    #             mask = mask.bool()
    #             tmp_batch_ids.masked_fill_(mask, value=torch.tensor(self.tokenizer.mask_token_id))
    #             mask_batch_ids.append(tmp_batch_ids)

    #     return torch.stack(mask_batch_ids, dim=0)

    def mask_sentence(self, sentence: str, probs: List[float]) -> List[str]:
        sentence_split = sentence.split()
        mask_nums = round(len(sentence_split) * self.mask_rate)
        if probs is not None:
            assert len(sentence_split) == len(probs)
        tmp_sentences = []
        for _ in range(self.mask_ensemble_numbers):
            tmp_sentence_list = copy(sentence_split)
            mask_token_ids = np.random.choice(list(range(len(sentence_split))),size=(mask_nums,),replace=False,p=probs)
            for index in mask_token_ids:
                tmp_sentence_list[index] = self.tokenizer.mask_token
            tmp_sentences.append(' '.join(tmp_sentence_list))
        return tmp_sentences

    def mask_batch_inputs(self, inputs: List[str], mask_probs: List[Union[List[float], None]]) -> List[str]:
        ensemble_sentence = []
        for sentence, mask_prob in zip(inputs, mask_probs):
            ensemble_sentence.extend(self.mask_sentence(sentence, mask_prob))
        return ensemble_sentence

    def predict_on_batch(self, ) -> torch.Tensor:
        

    @overrides
    def forward(
        self,
        inputs: List[str],
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        if self.random_mask:
            mask_probs = [None for _ in range(len(inputs))]
        else:
            # mask probs: List[List[float]], for each element, represents probs
            mask_probs = self.get_sentence_probs(inputs)
            
        mask_ensemble_inputs = self.mask_batch_inputs(inputs, mask_probs)
        tensors_input = self.batch_encode_plus(mask_ensemble_inputs)
        logits = self.classifier(**tensors_input)[0]
        split_logits = torch.split(logits, split_size_or_sections=self.mask_ensemble_numbers)
        label_numbers = logits.shape[-1]
        classifier_logits = []
        for logit in split_logits:
            tmp_logits = torch.bincount(torch.argmax(logit, dim=-1), minlength=label_numbers).float() / self.mask_ensemble_numbers
            classifier_logits. (tmp_logits)
        classifier_logits = torch.stack(classifier_logits)
        return (classifier_logits, )

