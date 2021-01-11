import os
import torch
import logging
import argparse
from overrides import overrides
from utils.config import DATASET_TYPE, PRETRAINED_MODEL_TYPE
from utils import check_and_create_path, set_seed

class ClassifierArgs:
    def __init__(self):
        super(ClassifierArgs, self).__init__()
        self.mode = 'predict' # in ['train', 'attack', 'evaluate', 'certify'
        self.file_name = None

        self.seed = 123
        # for evaluate and predict, default is 'test', meaning test set
        self.evaluation_data_type = 'test'

        self.dataset_name = 'agnews'
        self.dataset_dir = '/home/zjiehang/SparseNLP/dataset'
        self.model_type = 'roberta'
        self.model_name_or_path = '/home/zjiehang/SparseNLP/pretrained/roberta-base-english'

        # for processing data 
        self.max_seq_length = 128
        self.case = 'lower'

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
        self.saving_step = None  # saving step for epoch

        self.tensorboard = None

        # training loss type, only support 'freelb', 'conat'  'pgd',  'hotflip', 'metric', 'metric_token', 'sparse' now
        # default is None, meaning normal training
        self.training_type = 'sparse'

        # for sparse adversarial training in NLP
        self.sparse_mask_rate = 0.9

        # for predict on sparse NLP
        self.predict_ensemble = 1000
        self.predict_numbers = None # default is None, meaning all dataset is used to evaluate
        # for certify on sparse NLP
        self.cerity_ensemble = 10000
        self.certiy_numbers = None # default is None, meaning all dataset is used to evaluate
        # for confidence alpha probability
        self.alpha = 0.05

        # for attack
        self.attack_times = 1 # attack times for average record
        self.attack_method = 'textfooler' # attack algorithm
        self.attack_numbers = 1000 # the examples numbers to be attack
        self.attack_constraints = 'retain' # in ['delete', 'retain']

    def __str__(self):
        return "Args:\n{}\n".format("\n".join(
            ["\t--{}={}".format(key, str(value)) for key, value in self.__dict__.items()]
        ))

    @property
    def do_lower_case(self) -> bool:
        return True if self.case == 'lower' else False

    @property
    def remove_attack_constrainst(self) -> bool:
        return True if self.attack_constraints == 'delete' else False

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
        file_name = self.training_type if self.training_type is not None else "" 
        hyper_parameter_dict = {'len': self.max_seq_length, 'epo': self.epochs, 'batch': self.batch_size}
        # if self.training_type == 'conat' or self.training_type == 'circleat':
        #     hyper_parameter_dict['aug'] = self.augmentation_numbers
        #     hyper_parameter_dict['un'] = self.unrep_loss
        #     hyper_parameter_dict['sup'] = self.rep_loss
        #     hyper_parameter_dict['rate'] = self.attack_max_rate_for_training
        # elif self.training_type == 'freelb' or self.training_type == 'pgd':
        #     hyper_parameter_dict['advstep'] = self.adv_steps
        #     hyper_parameter_dict['advlr'] = self.adv_learning_rate
        #     hyper_parameter_dict['norm'] = self.adv_max_norm
        # elif self.training_type == 'hotflip':
        #     hyper_parameter_dict['rate'] = self.attack_max_rate_for_training
        #     hyper_parameter_dict['advstep'] = self.adv_steps
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

    @classmethod
    def parse(cls, verbose=False) -> "ClassifierArgs":
        parser = argparse.ArgumentParser()
        default_args = cls()
        for key, value in default_args.__dict__.items():
            if type(value) == bool:
                raise Exception("Bool value is not supported!!!")
            parser.add_argument('--{}'.format(key),
                                action='store',
                                default=value,
                                type=type(value) if value is not None else str,
                                dest=str(key))
        parsed_args = parser.parse_args(namespace=default_args)
        if parsed_args.mode !='infer' and verbose:
            print("Args:")
            for key, value in parsed_args.__dict__.items():
                print("\t--{}={}".format(key, value))

        # change model type and dataset type to lower case
        parsed_args.model_type = parsed_args.model_type.lower()
        parsed_args.dataset_name = parsed_args.dataset_name.lower()

        assert isinstance(parsed_args, ClassifierArgs)
        return parsed_args