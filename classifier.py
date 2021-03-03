# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import math
import torch
import logging
import numpy as np
import torch.nn as nn
from overrides import overrides
from typing import List, Any, Dict, Union, Tuple
from tqdm import tqdm
from torchnlp.samplers.bucket_batch_sampler import BucketBatchSampler
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizer

from args import ClassifierArgs
from utils.config import PRETRAINED_MODEL_TYPE, DATASET_TYPE
from data.reader import DataReader
from data.processor import DataProcessor
from data.instance import InputInstance
from data.dataset import ListDataset
from utils.metrics import Metric, RandomSmoothAccuracyMetrics, RandomAblationCertifyMetric
from utils.loss import ContrastiveLearningLoss, UnsupervisedCircleLoss
from utils.mask import mask_instance, mask_forbidden_index
from predictor import Predictor
from utils.utils import collate_fn, xlnet_collate_fn, convert_batch_to_bert_input_dict, build_forbidden_mask_words
from utils.hook import EmbeddingHook
from trainer import (BaseTrainer,
                    FreeLBTrainer,
                    PGDTrainer,
                    HotflipTrainer,
                    EmbeddingLevelMetricTrainer,
                    TokenLevelMetricTrainer,
                    RepresentationLearningTrainer,
                    MaskTrainer,
                    SAFERTrainer
                    )
from utils.textattack import build_english_attacker
from utils.textattack import CustomTextAttackDataset, SimplifidResult
from textattack.models.wrappers import HuggingFaceModelWrapper, HuggingFaceModelMaskEnsembleWrapper, HuggingFaceModelSaferEnsembleWrapper
from textattack.loggers.attack_log_manager import AttackLogManager
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult
from utils.public import auto_create
from utils.certify import predict, lc_bound, population_radius_for_majority, population_radius_for_majority_by_estimating_lambda, population_lambda
from torch.optim.adamw import AdamW


class Classifier:
    def __init__(self, args: ClassifierArgs):
        # check mode
        self.methods = {'train': self.train, 
                        'evaluate': self.evaluate,
                        'predict': self.predict, 
                        'attack': self.attack,
                        'augmentation': self.augmentation,
                        'certify': self.certify,
                        'statistics': self.statistics
                        }# 'certify': self.certify}
        assert args.mode in self.methods, 'mode {} not found'.format(args.mode)

        # for data_reader and processing
        self.data_reader, self.tokenizer, self.data_processor = self.build_data_processor(args)
        self.model = self.build_model(args)
        self.type_accept_instance_as_input = ['conat', 'sparse', 'safer']
        self.loss_function = self.build_criterion(args.dataset_name)
        
        self.forbidden_words = None
        if args.keep_sentiment_word:
            self.forbidden_words = build_forbidden_mask_words(args.sentiment_path)

    def save_model_to_file(self, save_dir: str, file_name: str):
        save_file_name = '{}.pth'.format(file_name)
        save_path = os.path.join(save_dir, save_file_name)
        torch.save(self.model.state_dict(), save_path)
        logging.info('Saving model to {}'.format(save_path))

    def loading_model_from_file(self, save_dir: str, file_name: str):
        load_file_name = '{}.pth'.format(file_name)
        load_path = os.path.join(save_dir, load_file_name)
        assert os.path.exists(load_path) and os.path.isfile(load_path), '{} not exits'.format(load_path)
        self.model.load_state_dict(torch.load(load_path), strict=False)
        logging.info('Loading model from {}'.format(load_path))

    def build_optimizer(self, args: ClassifierArgs, **kwargs):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        return optimizer

    def build_model(self, args: ClassifierArgs) -> nn.Module:
        # config_class: PreTrainedConfig
        # model_class: PreTrainedModel
        config_class, model_class, _ = PRETRAINED_MODEL_TYPE.MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(
            args.model_name_or_path,
            num_labels=self.data_reader.NUM_LABELS,
            finetuning_task=args.dataset_name,
            output_hidden_states=True,
        )
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool('ckpt' in args.model_name_or_path),
            config=config
        ).cuda()
        return model

    def build_data_processor(self, args: ClassifierArgs, **kwargs) -> List[Union[DataReader, PreTrainedTokenizer, DataProcessor]]:
        data_reader = DATASET_TYPE.DATA_READER[args.dataset_name]()
        _, _, tokenizer_class = PRETRAINED_MODEL_TYPE.MODEL_CLASSES[args.model_type]
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path,
            do_lower_case=args.do_lower_case
        )
        data_processor = DataProcessor(data_reader=data_reader,
                                       tokenizer=tokenizer,
                                       model_type=args.model_type,
                                       max_seq_length=args.max_seq_length)

        return [data_reader, tokenizer, data_processor]

    def build_criterion(self, dataset):
        return DATASET_TYPE.get_loss_function(dataset)

    def build_data_loader(self, args: ClassifierArgs, data_type: str, tokenizer: bool = True, **kwargs) -> List[Union[Dataset, DataLoader]]:
        # for some training type, when training, the inputs type is Inputstance
        if data_type == 'train' and args.training_type in self.type_accept_instance_as_input:
            tokenizer = False
        shuffle = True if data_type == 'train' else False
        file_name = data_type
        if file_name == 'train' and args.file_name is not None:
            file_name = args.file_name
        dataset = auto_create('{}_max{}{}'.format(file_name, args.max_seq_length, '_tokenizer' if tokenizer else ''),
                            lambda: self.data_processor.read_from_file(args.dataset_dir, file_name, tokenizer=tokenizer),
                            True, args.caching_dir)
        
        # for collate function
        if tokenizer:
            collate_function = xlnet_collate_fn if args.model_type in ['xlnet'] else collate_fn
        else:
            collate_function = lambda x: x
        
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_function)
        return [dataset, data_loader]


    def build_attacker(self, args: ClassifierArgs, **kwargs):
        if args.training_type == 'sparse' or args.training_type == 'safer':
            if args.dataset_name in ['agnews', 'imdb']:
                batch_size = 300
            else:
                batch_size = 600
            if args.training_type == 'sparse':
                model_wrapper = HuggingFaceModelMaskEnsembleWrapper(args, 
                                                                    self.model, 
                                                                    self.tokenizer, 
                                                                    batch_size=batch_size)
            else:
                model_wrapper = HuggingFaceModelSaferEnsembleWrapper(args, 
                                                                    self.model, 
                                                                    self.tokenizer, 
                                                                    batch_size=batch_size)
        else:
            model_wrapper = HuggingFaceModelWrapper(self.model, self.tokenizer, batch_size=args.batch_size)
        

        attacker = build_english_attacker(args, model_wrapper)
        return attacker

    def build_writer(self, args: ClassifierArgs, **kwargs) -> Union[SummaryWriter, None]:
        writer = None
        if args.tensorboard == 'yes':
            tensorboard_file_name = '{}-tensorboard'.format(args.build_logging_path())
            tensorboard_path = os.path.join(args.logging_dir, tensorboard_file_name)
            writer = SummaryWriter(tensorboard_path)
        return writer

    def build_trainer(self, args: ClassifierArgs, dataset: Dataset, data_loader: DataLoader) -> BaseTrainer:
        # get optimizer
        optimizer = self.build_optimizer(args)

        # get learning rate decay
        lr_scheduler = CosineAnnealingLR(optimizer, len(dataset) // args.batch_size * args.epochs)

        # get tensorboard writer
        writer = self.build_writer(args)

        trainer = BaseTrainer(data_loader, self.model, self.loss_function, optimizer, lr_scheduler, writer)
        if args.training_type == 'freelb':
            trainer = FreeLBTrainer(data_loader, self.model, self.loss_function, optimizer, lr_scheduler, writer)
        elif args.training_type == 'pgd':
            trainer = PGDTrainer(data_loader, self.model, self.loss_function, optimizer, lr_scheduler, writer)
        elif args.training_type == 'advhotflip':
            trainer = HotflipTrainer(args, self.tokenizer, data_loader, self.model, self.loss_function, optimizer, lr_scheduler, writer)
        elif args.training_type == 'metric':
            trainer = EmbeddingLevelMetricTrainer(data_loader, self.model, self.loss_function, optimizer, lr_scheduler, writer)
        elif args.training_type == 'metric_token':
            trainer = TokenLevelMetricTrainer(args, self.tokenizer, data_loader, self.model, self.loss_function, optimizer, lr_scheduler, writer)
        elif args.training_type == 'sparse':
            # trick = True if args.dataset_name in ['mr'] else False
            trainer = MaskTrainer(args, self.data_processor, data_loader, self.model, self.loss_function, optimizer, lr_scheduler, writer)
        elif args.training_type == 'safer':
            trainer = SAFERTrainer(args, self.data_processor, data_loader, self.model, self.loss_function, optimizer, lr_scheduler, writer)
        return trainer

    def train(self, args: ClassifierArgs):
        # get dataset
        dataset, data_loader = self.build_data_loader(args, 'train')

        # get trainer
        trainer = self.build_trainer(args, dataset, data_loader)

        best_metric = None
        for epoch_time in range(args.epochs):
            trainer.train_epoch(args, epoch_time)

            # saving model according to epoch_time
            self.saving_model_by_epoch(args, epoch_time)

            # evaluate model according to epoch_time
            metric = self.evaluate(args, is_training=True)

            # update best metric
            # if best_metric is None, update it with epoch metric directly, otherwise compare it with epoch_metric
            if best_metric is None or metric > best_metric:
                best_metric = metric
                self.save_model_to_file(args.saving_dir, args.build_saving_file_name(description='best'))
        
        if args.training_type == 'sparse' and args.incremental_trick and args.saving_last_epoch:
            self.save_model_to_file(args.saving_dir, args.build_saving_file_name(description='best'))
        self.evaluate(args)

    @torch.no_grad()
    def evaluate(self, args: ClassifierArgs, is_training=False) -> Metric:
        if is_training:
            logging.info('Using current modeling parameter to evaluate')
            data_type = 'dev'
        else:
            self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description='best'))
            data_type = args.evaluation_data_type
        self.model.eval()

        dataset, data_loader = self.build_data_loader(args, data_type)
        epoch_iterator = tqdm(data_loader)

        metric = DATASET_TYPE.get_evaluation_metric(args.dataset_name,compare_key=args.compare_key)
        for step, batch in enumerate(epoch_iterator):
            assert isinstance(batch[0], torch.Tensor)
            batch = tuple(t.cuda() for t in batch)
            golds = batch[3]
            inputs = convert_batch_to_bert_input_dict(batch, args.model_type)
            logits = self.model.forward(**inputs)[0]
            losses = self.loss_function(logits.view(-1, self.data_reader.NUM_LABELS), golds.view(-1))
            epoch_iterator.set_description('loss: {:.4f}'.format(torch.mean(losses)))
            metric(losses, logits, golds)

        print(metric)
        logging.info(metric)
        return metric

    @torch.no_grad()
    def infer(self, args: ClassifierArgs) -> Dict:
        content = args.content
        assert content is not None, 'in infer mode, parameter content cannot be None! '
        content = content.strip()
        assert content != '' and len(content) != 0, 'in infer mode, parameter content cannot be empty! '

        self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description='best'))
        self.model.eval()

        predictor = Predictor(self.model, self.data_processor, args.model_type)
        pred_probs = predictor.predict(content)
        pred_label = np.argmax(pred_probs)
        pred_label = self.data_reader.get_idx_to_label(pred_label)
        if pred_label == '100':
            pred_label = '0'
        elif pred_label == '101':
            pred_label = '1'

        result_in_dict = {'content': content, 'pred_label':pred_label, 'pred_confidence': pred_probs}
        result_in_str = ', '.join(['{}: {}'.format(key, value)
                                   if not isinstance(value, list)
                                   else '{}: [{}]'.format(key, ', '.join(["%.4f" % val for val in value]))
                                   for key, value in result_in_dict.items()])
        print(result_in_str)
        logging.info(result_in_str)
        return result_in_dict

    # for sparse adversarial training with random mask,
    # predict() is to get the smoothing result, which is different from evaluate()
    @torch.no_grad()
    def predict(self, args: ClassifierArgs, **kwargs):
        # self.evaluate(args, is_training=False)
        self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description='best'))
        self.model.eval()
        predictor = Predictor(self.model, self.data_processor, args.model_type)

        dataset, _ = self.build_data_loader(args, args.evaluation_data_type, tokenizer=False)
        assert isinstance(dataset, ListDataset)
        if args.predict_numbers == -1:
            predict_dataset = dataset.data
        else:
            predict_dataset = np.random.choice(dataset.data, size=(args.predict_numbers, ), replace=False)
        
        description = tqdm(predict_dataset)
        metric = RandomSmoothAccuracyMetrics()
        for data in description:
            tmp_instances = self.mask_instance_decorator(args, data, args.predict_ensemble)
            tmp_probs = predictor.predict_batch(tmp_instances)
            target = self.data_reader.get_label_to_idx(data.label)
            pred = predict(tmp_probs, args.alpha)
            metric(pred, target)
            description.set_description(metric.__str__())
        print(metric)
        logging.info(metric)
    
    def attack(self, args: ClassifierArgs, **kwargs):
        # self.evaluate(args, is_training=False)
        # self.evaluate(args, is_training=False)
        self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description='best'))
        self.model.eval()

        # build test dataset 
        dataset, _ = self.build_data_loader(args, args.evaluation_data_type, tokenizer=False)
        test_instances = dataset.data
       
        # build attacker
        attacker = self.build_attacker(args)

        attacker_log_path = '{}'.format(args.build_logging_path())
        attacker_log_path = os.path.join(args.logging_dir, attacker_log_path)
        attacker_log_manager = AttackLogManager()
        # attacker_log_manager.enable_stdout()
        attacker_log_manager.add_output_file(os.path.join(attacker_log_path, '{}.txt'.format(args.attack_method)))
        
        for i in range(args.attack_times):
            print("Attack time {}".format(i))
            
            choice_instances = np.random.choice(test_instances, size=(args.attack_numbers,),replace=False)
            dataset = CustomTextAttackDataset.from_instances(args.dataset_name, choice_instances, self.data_reader.get_labels())
            results_iterable = attacker.attack_dataset(dataset)
            description = tqdm(results_iterable, total=len(choice_instances))
            result_statistics = SimplifidResult()
            for result in description:
                try:
                    attacker_log_manager.log_result(result)
                    result_statistics(result)
                    description.set_description(result_statistics.__str__())
                except RuntimeError as e:
                    print('error in process')

        attacker_log_manager.enable_stdout()
        attacker_log_manager.log_summary()

    def augmentation(self, args: ClassifierArgs, **kwargs):
        self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description='best'))
        self.model.eval()

        train_instances, _ = self.build_data_loader(args, 'train', tokenizer=False)
        train_dataset_len = len(train_instances.data)
        print('Training Set: {} sentences. '.format(train_dataset_len))
        
        # delete instance whose length is smaller than 3
        train_instances_deleted = [instance for instance in train_instances.data if instance.length() >= 3]
        dataset_to_aug = np.random.choice(train_instances_deleted, size=(int(train_dataset_len * 0.5), ), replace=False)

        dataset_to_write = np.random.choice(train_instances.data, size=(int(train_dataset_len * 0.5), ), replace=False).tolist()
        attacker = self.build_attacker(args)
        attacker_log_manager = AttackLogManager()
        dataset = CustomTextAttackDataset.from_instances(args.dataset_name, dataset_to_aug, self.data_reader.get_labels())
        results_iterable = attacker.attack_dataset(dataset)
        aug_instances = []
        for result, instance in tqdm(zip(results_iterable, dataset_to_aug), total=len(dataset)):
            try:
                adv_sentence = result.perturbed_text()
                aug_instances.append(InputInstance.from_instance_and_perturb_sentence(instance, adv_sentence))
            except:
                print('one error happend, delete one instance')

        dataset_to_write.extend(aug_instances)
        self.data_reader.saving_instances(dataset_to_write, args.dataset_dir, 'aug_{}'.format(args.attack_method))
        print('Writing {} Sentence. '.format(len(dataset_to_write)))
        attacker_log_manager.enable_stdout()
        attacker_log_manager.log_summary()


    def certify(self, args: ClassifierArgs, **kwargs):
        # self.evaluate(args, is_training=False)
        self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description='best'))
        self.model.eval()
        predictor = Predictor(self.model, self.data_processor, args.model_type)

        dataset, _ = self.build_data_loader(args, args.evaluation_data_type, tokenizer=False)
        assert isinstance(dataset, ListDataset)
        if args.certify_numbers == -1:
            certify_dataset = dataset.data
        else:
            certify_dataset = np.random.choice(dataset.data, size=(args.certify_numbers, ), replace=False)
        
        description = tqdm(certify_dataset)
        num_labels = self.data_reader.NUM_LABELS
        metric = RandomAblationCertifyMetric() 
        for data in description:
            target = self.data_reader.get_label_to_idx(data.label)
            data_length = data.length()
            keep_nums = data_length - round(data_length * args.sparse_mask_rate)

            tmp_instances = self.mask_instance_decorator(args, data, args.predict_ensemble)
            tmp_probs = predictor.predict_batch(tmp_instances)
            guess = np.argmax(np.bincount(np.argmax(tmp_probs, axis=-1), minlength=num_labels))

            if guess != target:
                metric(np.nan, data_length)
                continue
                
            tmp_instances = self.mask_instance_decorator(args, data, args.ceritfy_ensemble)
            tmp_probs = predictor.predict_batch(tmp_instances)
            guess_counts = np.bincount(np.argmax(tmp_probs, axis=-1), minlength=num_labels)[guess]
            lower_bound, upper_bound = lc_bound(guess_counts, args.ceritfy_ensemble, args.alpha)
            if args.certify_lambda:
                # tmp_instances, mask_indexes = mask_instance(data, args.sparse_mask_rate, self.tokenizer.mask_token,nums=args.ceritfy_ensemble * 2, return_indexes=True)
                # tmp_probs = predictor.predict_batch(tmp_instances)
                # tmp_preds = np.argmax(tmp_probs, axis=-1)
                # ablation_indexes = [list(set(list(range(data_length))) - set(indexes.tolist())) for indexes in mask_indexes]
                # radius = population_radius_for_majority_by_estimating_lambda(lower_bound, data_length, keep_nums, tmp_preds, ablation_indexes, num_labels, guess, samplers = 200)
                radius = population_radius_for_majority(lower_bound, data_length, keep_nums, lambda_value=guess_counts / args.ceritfy_ensemble)
            else:
                radius = population_radius_for_majority(lower_bound, data_length, keep_nums)
            
            metric(radius, data_length)

            result = metric.get_metric()
            description.set_description("Accu: {:.2f}%, Median: {}".format(result['accuracy'] * 100, result['median']))
        print(metric)
        logging.info(metric)

        # logging metric certify_radius and length
        logging.info(metric.certify_radius())
        logging.info(metric.sentence_length())


    def statistics(self, args: ClassifierArgs, **kwargs):
        # self.evaluate(args, is_training=False)
        self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description='best'))
        self.model.eval()
        predictor = Predictor(self.model, self.data_processor, args.model_type)

        dataset, _ = self.build_data_loader(args, args.evaluation_data_type, tokenizer=False)
        assert isinstance(dataset, ListDataset)
        if args.certify_numbers == -1:
            certify_dataset = dataset.data
        else:
            certify_dataset = np.random.choice(dataset.data, size=(args.certify_numbers, ), replace=False)
        
        description = tqdm(certify_dataset)
        num_labels = self.data_reader.NUM_LABELS
        metric = RandomAblationCertifyMetric() 
        result_dicts = {"pix": []}
        for i in range(11):
            result_dicts[str(i)] = list()
        for data in description:
            target = self.data_reader.get_label_to_idx(data.label)
            data_length = data.length()
            keep_nums = data_length - round(data_length * args.sparse_mask_rate)

            tmp_instances = self.mask_instance_decorator(args, data, args.predict_ensemble)
            tmp_probs = predictor.predict_batch(tmp_instances)
            guess = np.argmax(np.bincount(np.argmax(tmp_probs, axis=-1), minlength=num_labels))

            if guess != target:
                metric(np.nan, data_length)
                continue

            numbers = args.ceritfy_ensemble * 2                
            tmp_instances, mask_indexes = self.mask_instance_decorator(args, data, numbers, return_indexes=True)
            ablation_indexes = [list(set(list(range(data_length))) - set(indexes)) for indexes in mask_indexes]
            tmp_probs = predictor.predict_batch(tmp_instances)
            tmp_preds = np.argmax(tmp_probs, axis=-1)
            p_i_x = np.bincount(tmp_preds, minlength=num_labels)[guess] / numbers
            result_dicts["pix"].append(p_i_x)
            for i in range(1, 11):
                lambda_value = population_lambda(tmp_preds, ablation_indexes, data_length, i, num_labels, guess)
                result_dicts[str(i)].append(lambda_value)
        
        file_name = os.path.join(args.logging_dir, "{}-probs.txt".format(args.build_logging_path()))
        with open(file_name, 'w') as file:
            for key, value in result_dicts.items():
                file.write(key)
                file.write(":  ")
                file.write(" ".join([str(v) for v in value]))
                file.write("\n")

    def saving_model_by_epoch(self, args: ClassifierArgs, epoch: int):
        # saving
        if args.saving_step is not None and args.saving_step != 0:
            if (epoch - 1) % args.saving_step == 0:
                self.save_model_to_file(args.saving_dir,
                                        args.build_saving_file_name(description='epoch{}'.format(epoch)))


    def mask_instance_decorator(self, args: ClassifierArgs, instance:InputInstance, numbers:int=1, return_indexes:bool=False):
        if self.forbidden_words is not None:
            forbidden_index = mask_forbidden_index(instance.perturbable_sentence(), self.forbidden_words)
            return mask_instance(instance, args.sparse_mask_rate, self.tokenizer.mask_token, numbers, return_indexes, forbidden_index)
        else:
            return mask_instance(instance, args.sparse_mask_rate, self.tokenizer.mask_token, numbers, return_indexes)


    @classmethod
    def run(cls, args: ClassifierArgs):
        # build logging
        # including check logging path, and set logging config
        args.build_logging_dir()
        args.build_logging()
        logging.info(args)

        args.build_environment()
        # check dataset and its path
        args.build_dataset_dir()

        args.build_saving_dir()
        args.build_caching_dir()

        if args.dataset_name in ['agnews', 'snli']:
            args.keep_sentiment_word = False

        classifier = cls(args)
        classifier.methods[args.mode](args)