# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import math
import torch
import logging
import numpy as np
import torch.nn as nn
from overrides import overrides
from typing import List, Any, Dict, Union
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
from utils.metrics import Metric, RandomSmoothAccuracyMetrics
from utils.loss import ContrastiveLearningLoss, UnsupervisedCircleLoss
from utils.mask import mask_instance
from predictor import Predictor
from utils.utils import collate_fn, xlnet_collate_fn, convert_batch_to_bert_input_dict
from utils.hook import EmbeddingHook
from trainer import (BaseTrainer,
                    FreeLBTrainer,
                    PGDTrainer,
                    HotflipTrainer,
                    EmbeddingLevelMetricTrainer,
                    TokenLevelMetricTrainer,
                    RepresentationLearningTrainer,
                    MaskTrainer)
from utils.textattack import build_english_attacker
from utils.textattack import CustomTextAttackDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.loggers.attack_log_manager import AttackLogManager
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult
from utils.public import auto_create
from torch.optim.adamw import AdamW


class Classifier:
    def __init__(self, args: ClassifierArgs):
        # check mode
        self.methods = {'train': self.train, 
                        'evaluate': self.evaluate,
                        'predict': self.predict, 
                        'attack': self.attack,
                        'augmentation': self.augmentation,
                        }# 'certify': self.certify}
        assert args.mode in self.methods, 'mode {} not found'.format(args.mode)

        # for data_reader and processing
        self.data_reader, self.tokenizer, self.data_processor = self.build_data_processor(args)
        self.model = self.build_model(args)
        self.type_accept_instance_as_input = ['conat', 'sparse']
        self.loss_function = self.build_criterion(args.dataset_name)

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

    def build_dataset(self, args: ClassifierArgs, data_type: str, tokenizer: bool = True) -> Dataset:
        # for some training type, when training, the inputs type is Inputstance
        if data_type == 'train' and args.training_type in self.type_accept_instance_as_input:
            tokenizer = False
        
        file_name = data_type if args.file_name is None else args.file_name
        dataset = auto_create('{}_max{}{}'.format(file_name, args.max_seq_length, '_tokenizer' if tokenizer else ''),
                            lambda: self.data_processor.read_from_file(args.dataset_dir, data_type, tokenizer=tokenizer))
        return dataset

    def build_data_loader(self, args: ClassifierArgs, data_type: str, tokenizer: bool = True, **kwargs) -> List[Union[Dataset, DataLoader]]:
        # for some training type, when training, the inputs type is Inputstance
        if data_type == 'train' and args.training_type in self.type_accept_instance_as_input:
            tokenizer = False
        shuffle = True if data_type == 'train' else False
        file_name = data_type if args.file_name is None else args.file_name
        dataset = auto_create('{}_max{}{}'.format(file_name, args.max_seq_length, '_tokenizer' if tokenizer else ''),
                            lambda: self.data_processor.read_from_file(args.dataset_dir, data_type, tokenizer=tokenizer),
                            True, args.caching_dir)
        
        # for collate function
        if tokenizer:
            collate_function = xlnet_collate_fn if args.model_type in ['xlnet'] else collate_fn
        else:
            collate_function = lambda x: x
        
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_function)
        return [dataset, data_loader]


    def build_attacker(self, args: ClassifierArgs, **kwargs):
        model_wrapper = HuggingFaceModelWrapper(self.model, self.tokenizer, batch_size=args.batch_size)
        self.model_wrapper = model_wrapper
        return build_english_attacker(args, model_wrapper)

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
        elif args.training_type == 'hotflip':
            trainer = HotflipTrainer(args, self.tokenizer, data_loader, self.model, self.loss_function, optimizer, lr_scheduler, writer)
        elif args.training_type == 'metric':
            trainer = EmbeddingLevelMetricTrainer(data_loader, self.model, self.loss_function, optimizer, lr_scheduler, writer)
        elif args.training_type == 'metric_token':
            trainer = TokenLevelMetricTrainer(args, self.tokenizer, data_loader, self.model, self.loss_function, optimizer, lr_scheduler, writer)
        elif args.training_type == 'sparse':
            trainer = MaskTrainer(self.data_processor, data_loader, self.model, self.loss_function, optimizer, lr_scheduler, writer)
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
        if args.predict_numbers is None:
            predict_dataset = dataset.data
        else:
            predict_dataset = np.random.choice(dataset.data, size=(args.predict_numbers, ), replace=False)
        
        description = tqdm(predict_dataset)
        metric = RandomSmoothAccuracyMetrics()
        for data in description:
            tmp_instances = mask_instance(data, args.sparse_mask_rate, self.tokenizer.mask_token,nums=args.predict_ensemble)
            tmp_probs = predictor.predict_batch(tmp_instances)
            label_in_int = self.data_reader.get_label_to_idx(data.label)
            metric(tmp_probs, label_in_int, args.alpha)
            description.set_description(metric.__str__())
        print(metric)
        logging.info(metric)
    
    def attack(self, args: ClassifierArgs, **kwargs):
        # self.evaluate(args, is_training=False)
        self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description='best'))

        # predictor = Predictor(self.model, self.data_processor, args.model_type)
        attacker_log_path = '{}'.format(args.build_logging_path())
        attacker_log_path = os.path.join(args.logging_dir, attacker_log_path)
        attacker = self.build_attacker(args)
        test_instances = self.data_reader.get_instances(args.dataset_dir, args.evaluation_data_type, 'dev')
        
        results = self.model_wrapper([instance.text_a for instance in test_instances])
        pred_list = np.argmax(results, axis=1).tolist()
        gold_list = [self.data_reader.get_label_to_idx(instance.label) for instance in test_instances]
        pred_true_list = [index for index, (pred, gold) in enumerate(zip(pred_list, gold_list)) if pred == gold]
        print("Accuracy: {:.2f}%".format(len(pred_true_list) * 100.0 / len(gold_list)))
        logging.info("Accuracy: {:.2f}%".format(len(pred_true_list) * 100.0 / len(gold_list)))


        is_nli_task = True if test_instances[0].text_b is not None else False

        attacker_log_manager = AttackLogManager()
        # attacker_log_manager.enable_stdout()
        attacker_log_manager.add_output_file(os.path.join(attacker_log_path, '{}.txt'.format(args.attack_method)))
        for i in range(args.attack_times):
            print("Attack time {}".format(i))
                       
            choice_instances = np.random.choice(test_instances, size=(args.attack_numbers,),replace=False)
            if is_nli_task:
                dataset = [[instance.text_a, instance.text_b, self.data_reader.get_label_to_idx(instance.label)] for instance in choice_instances]
            else:
                dataset = [(instance.text_a, self.data_reader.get_label_to_idx(instance.label)) for instance in choice_instances]
            # dataset = [InputInstance("None", instance.text_a, 
            #                         instance.text_b,
            #                         self.data_reader.get_label_to_idx(instance.label)) for instance in choice_instances]
            dataset = CustomTextAttackDataset.from_instances(args.dataset_name, dataset,self.data_reader.get_labels())

            results_iterable = attacker.attack_dataset(dataset)
            num_successes = 0
            num_failures = 0
            all_numbers = 0
            description = tqdm(results_iterable, total=len(choice_instances))
            for result in description:
                try:
                    attacker_log_manager.log_result(result)
                    if isinstance(result, SuccessfulAttackResult):
                        num_successes += 1
                    elif isinstance(result, FailedAttackResult):
                        num_failures += 1
                    all_numbers += 1

                    pred_correct_nums = num_successes + num_failures
                    if pred_correct_nums == 0:
                        description.set_description('Succ Rate:{:.2f}%, Accu: {:.2f}%'.format(0.0, num_failures / all_numbers * 100))
                    else:
                        description.set_description('Succ Rate:{:.2f}%, Accu: {:.2f}%'.format(num_successes / pred_correct_nums * 100, num_failures / all_numbers * 100))
                except RuntimeError as e:
                    print('error in process')

        attacker_log_manager.enable_stdout()
        attacker_log_manager.log_summary()

    def augmentation(self, args: ClassifierArgs, **kwargs):
        pass
        # self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description='best'))
        # self.model.eval()

        # train_instances = self.data_reader.get_instances(args.dataset_dir, 'train', 'train')
        # print('Training Set: {} sentences. '.format(len(train_instances)))


        # model = get_model(args.tokenizer_path,args.model_path, args.dataset)
        # attacker = get_attacker(model, args.running_type, args.attacker_type)
        # attacker_log_manager = AttackLogManager()
        # writing_numbers = 0
        # with open(args.augmentation_file_path, 'w', encoding='utf8') as file:
        #     results_iterable = attacker.attack_dataset(dataset)
        #     for dataset, result in tqdm(zip(dataset, results_iterable),total=len(dataset)):
        #         clean_sentence = result.original_text()
        #         adv_sentence = result.perturbed_text()
        #         label = dataset[1]
        #         file.write('{}\t{}\n'.format(clean_sentence, label))
        #         writing_numbers += 1
        #         file.write('{}\t{}\n'.format(adv_sentence, label))
        #         writing_numbers += 1
        # print('Writing {} Sentence to {}'.format(writing_numbers, args.augmentation_file_path))
        # attacker_log_manager.enable_stdout()
        # attacker_log_manager.log_summary()

        # predictor = Predictor(self.model, self.data_processor, args.)
        # attacker = self.build_attacker(args, predictor)

        # train_instances = self.data_reader.get_instances(args.dataset_dir, 'train', 'train')
        # aug_instances = []
        # for instance in tqdm(train_instances):
        #     aug_instance = attacker.augment(instance)
        #     if aug_instance is not None:
        #         aug_instance.set_guid('{}-aug'.format(instance.guid))
        #         aug_instances.append(instance)
        #         aug_instances.append(aug_instance)
        #     else:
        #         aug_instances.append(instance)
        # self.data_reader.saving_instances(aug_instances, args.dataset_dir, 'aug')
        # print('saving {} augmentation data successfully to {}'.format(len(aug_instances), os.path.join(args.dataset_dir, 'aug')))
        # logging.info('saving {} augmentation data successfully to {}'.format(len(aug_instances), os.path.join(args.dataset_dir, 'aug')))


    def certify(self, args: ClassifierArgs, **kwargs):
        pass

    def saving_model_by_epoch(self, args:ClassifierArgs, epoch: int):
        # saving
        if args.saving_step is not None and args.saving_step != 0:
            if (epoch - 1) % args.saving_step == 0:
                self.save_model_to_file(args.saving_dir,
                                        args.build_saving_file_name(description='epoch{}'.format(epoch)))


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

        classifier = cls(args)
        classifier.methods[args.mode](args)