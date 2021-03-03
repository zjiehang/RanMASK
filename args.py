import os
import torch
import logging
import argparse
from overrides import overrides
from utils.config import DATASET_TYPE, PRETRAINED_MODEL_TYPE
from utils import check_and_create_path, set_seed
from baseargs import ProgramArgs

class ClassifierArgs(ProgramArgs):
    def __init__(self):
        super(ClassifierArgs, self).__init__()
        # in ['train', 'attack', 'evaluate', 'certify', 'statistics']
        self.mode = 'attack' 
        self.file_name = None

        self.seed = None
        # for evaluate and predict, default is 'test', meaning test set
        self.evaluation_data_type = 'test'

        self.dataset_name = 'agnews'
        self.dataset_dir = '/home/zjiehang/SparseNLP/dataset'
        self.model_type = 'roberta'
        self.model_name_or_path = '/home/zjiehang/SparseNLP/pretrained/roberta-base-english'

        # for processing data 
        self.max_seq_length = 128
        self.do_lower_case = True

        # base training hyper-parameters, if need other, define in subclass
        self.epochs = 10  # training epochs
        self.batch_size = 32  # batch size
        self.gradient_accumulation_steps = 1  # Number of updates steps to accumulate before performing a backward/update pass.
        self.learning_rate = 5e-5  # The initial learning rate for Adam.
        self.weight_decay = 1e-6  # weight decay
        self.adam_epsilon = 1e-8  # epsilon for Adam optimizer
        self.max_grad_norm = 1.0  # max gradient norm
        self.learning_rate_decay = 0.1  # Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training

        # the key to compare when choosing the best modeling to be saved, default is '-loss'
        # where '+X' means the larger the X is, the better the modeling.
        # where '-X' means the smaller the X is, the better the modeling.
        # e.g., when X == '-loss', using loss to compare which epoch is best
        self.compare_key = '+accuracy'

        # saving, logging and caching
        self.caching_dir = '/home/zjiehang/SparseNLP/cache_path'  # for cache path
        self.saving_dir = '/home/zjiehang/SparseNLP/save_models'  # for saving modeling
        self.logging_dir = '/home/zjiehang/SparseNLP/result_log'  # for logging
        self.saving_step = 0  # saving step for epoch

        self.tensorboard = None

        # training loss type, only support 'freelb', 'conat'  'pgd',  'hotflip', 'metric', 'metric_token', 'sparse', 'safer' now
        # default is None, meaning normal training
        self.training_type = 'sparse'

        # for sparse adversarial training in NLP
        self.sparse_mask_rate = 0.7

        # for predict on sparse NLP
        self.predict_ensemble = 100
        self.predict_numbers = 1000 # default is None, meaning all dataset is used to evaluate
        # for certify on sparse NLP
        self.ceritfy_ensemble = 5000
        self.certify_numbers = 1000 # default is None, meaning all dataset is used to evaluate
        # whether to add lambda, for sparse NLP, 
        # pi(x)− Pr([f(ABLATE(x, T)) = i] ∧ [T ∩ (x diff with x')]) <= pi(x')
        # Pr([f(ABLATE(x, T)) = i] ∧ [T ∩ (x diff with x') != Ø]) = Pr([f(ABLATE(x, T)) = i] | [T ∩ (x diff with x')  != Ø]) * Pr(T ∩ (x diff with x')  != Ø)
        # where lambda = Pr([f(ABLATE(x, T)) = i] | [T ∩ (x diff with x')  != Ø]) * Pr(T ∩ (x diff with x')  != Ø)
        # if lambda is False, only use the delta Pr(T ∩ (x diff with x')  != Ø) for certificate robustness 
        self.certify_lambda = True
        # for confidence alpha probability
        self.alpha = 0.05

        # for attack
        self.attack_times = 1 # attack times for average record
        self.attack_method = 'textfooler' # attack algorithm
        self.attack_numbers = 100 # the examples numbers to be attack
        self.ensemble_method = 'votes' # in [votes mean]

        # for pgd-K and FreeLB (including adv-hotflip)
        self.adv_steps = 3 # Number of gradient ascent steps for the adversary, for FreeLB default 5
        self.adv_learning_rate = 3e-1 # Step size of gradient ascent, for FreeLB, default 0.03
        self.adv_init_mag = 5e-1 # Magnitude of initial (adversarial?) perturbation, for FreeLB, default 0.05
        self.adv_max_norm = 5e-2 # adv_max_norm = 0 means unlimited, for FreeLB, default 0.0
        self.adv_norm_type = 'l2' # norm type of the adversary
        self.adv_change_rate = 0.2 # rate for adv-hotflip, change rate of a sentence 

        # for sentiment-word file path
        self.sentiment_path = '/home/zjiehang/SparseNLP/dataset/sentiment_word/sentiment-words.txt'
        # when keep_sentiment_word, keep sentiment words when training
        self.keep_sentiment_word = False
        # whether to add incremental trick, the mask rate increases with the global training step increasing.
        # the default initial mask rate is 0.4
        self.incremental_trick = False
        self.initial_mask_rate = 0.6

        self.saving_last_epoch = False

        self.with_lm = False

        # perturbation set path for safer trainer
        self.safer_perturbation_set = 'perturbation_constraint_pca0.8_100.pkl'

        
    def build_environment(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.seed)

    def build_dataset_dir(self):
        assert self.dataset_name in DATASET_TYPE.DATA_READER.keys(), 'dataset not found {}'.format(self.dataset_name)
        testing_file = ['train.json', 'train.txt', 'train.csv', 'train.tsv']
        for file in testing_file:
            train_file_path = os.path.join(self.dataset_dir, file)
            if os.path.exists(train_file_path) and os.path.isfile(train_file_path):
                return
        self.dataset_dir = os.path.join(self.dataset_dir, self.dataset_name)
        for file in testing_file:
            train_file_path = os.path.join(self.dataset_dir, file)
            if os.path.exists(train_file_path) and os.path.isfile(train_file_path):
                return
        raise FileNotFoundError("Dataset file cannot be found in dir {}".format(self.dataset_dir))

    # setting new saving path
    # the new saving path is combined by
    # args.saving_dir = args.saving_dir/${data}_${model}
    def build_saving_dir(self):    
        self.saving_dir = os.path.join(self.saving_dir,  "{}_{}".format(self.dataset_name, self.model_type))
        check_and_create_path(self.saving_dir)

    def build_logging_dir(self):
        self.logging_dir = os.path.join(self.logging_dir, "{}_{}".format(self.dataset_name, self.model_type))
        check_and_create_path(self.logging_dir)

    def build_caching_dir(self):
        # build safer perturbation set path
        if self.safer_perturbation_set is not None:
            self.safer_perturbation_set = os.path.join(self.caching_dir, os.path.join(self.dataset_name, self.safer_perturbation_set))
        self.caching_dir = os.path.join(self.caching_dir, "{}_{}".format(self.dataset_name, self.model_type))
        check_and_create_path(self.caching_dir)

    def build_logging(self, **kwargs):
        logging_file_path = self.build_logging_file()
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=logging_file_path,level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


    def build_saving_file_name(self, description: str = None):
        '''
        build hyper-parameter for saving and loading model, some important hyper-parameters are set to define the saving file name
        :param args:
        :param description:
        :return:
        '''
        file_name = self.training_type
        if self.file_name is not None:
            file_name = "{}{}".format(file_name if file_name == "" else file_name+"_", self.file_name) 
        hyper_parameter_dict = {'len': self.max_seq_length, 'epo': self.epochs, 'batch': self.batch_size}
        if self.training_type == 'freelb' or self.training_type == 'pgd':
            hyper_parameter_dict['advstep'] = self.adv_steps
            hyper_parameter_dict['advlr'] = self.adv_learning_rate
            hyper_parameter_dict['norm'] = self.adv_max_norm
        elif self.training_type == 'advhotflip':
            hyper_parameter_dict['rate'] = self.adv_change_rate
            hyper_parameter_dict['advstep'] = self.adv_steps
        # elif self.training_type == 'metric' or self.training_type == 'metric_token':
        #     hyper_parameter_dict['rate'] = self.attack_max_rate_for_training
        #     hyper_parameter_dict['step'] = self.adv_steps
        #     hyper_parameter_dict['alpha'] = self.metric_learning_alpha
        #     hyper_parameter_dict['margin'] = self.metric_learning_margin
        if self.training_type == 'sparse':
            hyper_parameter_dict['rate'] = self.sparse_mask_rate

        if file_name == "":
            file_name = '{}'.format("-".join(["{}{}".format(key, value) for key, value in hyper_parameter_dict.items()]))
        else:
            file_name = '{}-{}'.format(file_name, "-".join(["{}{}".format(key, value) for key, value in hyper_parameter_dict.items()]))

        if description is not None:
            file_name = '{}-{}'.format(file_name, description)
        return file_name

    def build_logging_path(self):
        if self.mode is None:
            return self.build_saving_file_name()
        if self.mode == 'certify' and self.certify_lambda:
            return '{}-{}-{}'.format(self.mode, self.build_saving_file_name(), 'lambda')
        elif self.mode == 'attack':
            if self.training_type in ['sparse', 'safer']:
                logging_path = "{}-{}-{}".format(self.mode, self.build_saving_file_name(), self.ensemble_method)
                if self.with_lm:
                    logging_path = "{}-{}".format(logging_path, 'lm')
            else:
                logging_path = "{}-{}".format(self.mode, self.build_saving_file_name())
            return logging_path
        else:
            return '{}-{}'.format(self.mode, self.build_saving_file_name())

    def build_logging_file(self):
        if self.mode == 'attack':
            logging_path = self.build_logging_path()
            logging_path = os.path.join(self.logging_dir, logging_path)
            if not os.path.exists(logging_path):
                os.makedirs(logging_path)
            return os.path.join(logging_path, 'running.log')
        else:
            return os.path.join(self.logging_dir, '{}.log'.format(self.build_logging_path()))
