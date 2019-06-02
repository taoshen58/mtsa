import platform
import argparse
import time
import os
from os.path import join
from src.utils.file import mp_join
from src.utils.hparams import HParams, merge_params, print_params, underline_to_camel
import logging as log


class Configs(object):
    def __init__(self, params_center):
        # add default and parsed parameters to cfg
        self.params_center = params_center
        self.dataset_dir = "./dataset"
        self.project_dir = "./"

        self.processed_name = self.get_params_str(
            ['dataset']
        ) + '_proprec.pickle'

        if self['network_type'] is None or self['network_type'] == 'test':
            self.model_name = '_test'
        else:
            model_name_params = ['dataset', 'network_class', 'network_type', 'lr', 'n_steps']
            if self['model_class'] is not None:
                model_name_params += self['model_class'].get_identity_param_list()
            else:
                print('fatal error: can not reach the model class')
            self.model_name = self.get_params_str(model_name_params)

        self.ckpt_name = 'model_file.ckpt'
        self.log_name = 'log_' + Configs.time_suffix() + '.txt'

        if self['dataset'] == 'snli':
            data_name_pattern = 'snli_1.0_%s.jsonl'
            self.raw_data_dir = join(self.dataset_dir, 'snli_1.0')
        elif self['dataset'].startswith('multinli'):
            self.raw_data_dir = join(self.dataset_dir, 'multinli_1.0')
            if self['dataset'] == 'multinli_m':
                data_name_pattern = 'multinli_1.0_%s_matched.jsonl'
            elif self['dataset'] == 'multinli_mm':
                data_name_pattern = 'multinli_1.0_%s_mismatched.jsonl'
            else:
                raise AttributeError
        else:
            raise AttributeError
        self.train_data_name, self.dev_data_name, self.test_data_name = \
            [data_name_pattern % name for name in ['train', 'dev', 'test']]
        # -------  dir -------
        self.bpe_data_dir = join(self.dataset_dir, 'bpe')
        self.pretrained_transformer_dir = join(self.dataset_dir, 'pretrained_transformer')

        #
        self.runtime_dir = mp_join(self.project_dir, 'runtime')
        self.run_model_dir = mp_join(self.runtime_dir, 'run_model')
        self.processed_dir = mp_join(self.runtime_dir, 'preproc')

        self.cur_run_dir = mp_join(self.run_model_dir, self['model_dir_prefix'] + self.model_name)
        self.log_dir = mp_join(self.cur_run_dir, 'log_files')
        self.summary_dir = mp_join(self.cur_run_dir, 'summary')
        self.ckpt_dir = mp_join(self.cur_run_dir, 'ckpt')
        self.other_dir = mp_join(self.cur_run_dir, 'other')

        # path
        self.train_data_path = join(self.raw_data_dir, self.train_data_name)
        self.dev_data_path = join(self.raw_data_dir, self.dev_data_name)
        self.test_data_path = join(self.raw_data_dir, self.test_data_name)


        self.processed_path = join(self.processed_dir, self.processed_name)
        self.ckpt_path = join(self.ckpt_dir, self.ckpt_name)
        self.log_path = join(self.log_dir, self.log_name)

        # merge the paths to params
        path_params = HParams(
            train_data_path=self.train_data_path,
            dev_data_path=self.dev_data_path,
            test_data_path=self.test_data_path,
            bpe_data_dir=self.bpe_data_dir,
            pretrained_transformer_dir=self.pretrained_transformer_dir,
            runtime_dir=self.runtime_dir,
            run_model_dir=self.run_model_dir,
            processed_dir=self.processed_dir,
            cur_run_dir=self.cur_run_dir,
            log_dir=self.log_dir,
            summary_dir=self.summary_dir,
            ckpt_dir=self.ckpt_dir,
            other_dir=self.other_dir,
            # paths
            processed_path=self.processed_path,
            ckpt_path=self.ckpt_path,
            log_path=self.log_path,
        )
        self.params_center.register_hparams(path_params, 'path_params')

        # logging setup
        log.basicConfig(format='%(asctime)s: %(message)s', level=log.INFO, datefmt='%m/%d %I:%M:%S %p')
        file_handler = log.FileHandler(self.log_path)  # add a file handler to a logger
        log.getLogger().addHandler(file_handler)

        # other
        # cuda support
        os.environ['CUDA_VISIBLE_DEVICES'] = '' if self['gpu'].lower() == 'none' else self['gpu']
        # import torch
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #
        other_params = HParams(
            intX='int32',
            floatX='float32',
            # device=device
        )
        self.params_center.register_hparams(other_params, 'other_params')

        log.info(print_params(self.params, print_std=False))

    def get_params_str(self, params):
        assert self.params_center is not None

        def abbreviation(name):
            words = name.strip().split('_')
            abb = ''
            for word in words:
                abb += word[0]
            return abb

        abbreviations = map(abbreviation, params)
        model_params_str = ''
        for paramsStr, abb in zip(params, abbreviations):
            model_params_str += '_' + abb + '.' + str(eval('self[\'%s\']' % paramsStr))
        return model_params_str

    @staticmethod
    def time_suffix():
        return '-'.join(time.asctime(time.localtime(time.time())).split()[1:-1]).replace(':', '-')

    @property
    def params(self):
        return self.params_center.all_params

    def __getitem__(self, item):
        return self.params_center[item]


class ParamsCenter(object):
    def __init__(self):
        self.hparam_name_list = []
        # ------parsing input arguments"--------
        parser = argparse.ArgumentParser()
        parser.register('type', 'bool', (lambda x: x.lower() in ("yes", "true", "t", "1")))
        parser.add_argument('--mode', type=str, default='train', help='train_tasks')
        parser.add_argument('--dataset', type=str, default='snli', help='[snli|multinli_m|multinli_mm]')
        parser.add_argument('--network_class', type=str, default='transformer', help='None')
        parser.add_argument('--network_type', type=str, default=None, help='None')
        parser.add_argument('--gpu', type=str, default='3', help='selected gpu index')
        parser.add_argument('--gpu_mem', type=float, default=None, help='selected gpu index')
        parser.add_argument('--model_dir_prefix', type=str, default='prefix', help='model dir name prefix')
        parser.add_argument('--aws', type='bool', default=False, help='using aws')

        # parsing parameters group
        parser.add_argument('--preprocessing_params', type=str, default='', help='')
        parser.add_argument('--model_params', type=str, default='', help='')
        parser.add_argument('--training_params', type=str, default='', help='')

        parser.set_defaults(shuffle=True)
        args = parser.parse_args()
        self.parsed_params = HParams()
        for key, val in args.__dict__.items():
            self.parsed_params.add_hparam(key, val)
        self.register_hparams(self.parsed_params, 'parsed_params')

        # pre-processed
        self.preprocessed_params = self.get_default_preprocessing_params()
        self.preprocessed_params.parse(self.parsed_params.preprocessing_params)
        self.register_hparams(self.preprocessed_params, 'preprocessed_params')

        # model
        self.model_params = merge_params(
            self.get_default_model_parameters(),
            self.get_default_specific_model_parameters(
                self.parsed_params.network_class, self.parsed_params.network_type)
        )
        self.model_params.parse(self.parsed_params.model_params)
        self.register_hparams(self.model_params, 'model_params')

        # traning
        self.training_params = self.get_default_training_params()
        self.training_params.parse(self.parsed_params.training_params)
        self.register_hparams(self.training_params, 'training_params')

    @staticmethod
    def get_default_preprocessing_params():
        params = HParams(
            max_sent_len=50,
            load_preproc=True,
        )
        return params

    @staticmethod
    def get_default_model_parameters():
        return HParams(
        )

    @staticmethod
    def get_default_training_params():
        hparams = HParams(
            optimizer='openai_adam',
            grad_norm=1.,

            n_steps=90000,
            lr=6.25e-5,

            # control
            save_model=False,
            save_num=3,
            load_model=False,
            load_path='',

            summary_period=1000,
            eval_period=500,

            train_batch_size=20,
            test_batch_size=24,

        )
        return hparams

    @staticmethod
    def get_default_specific_model_parameters(network_class, network_type):
        model_params = HParams(
            model_class=None
        )
        if network_type is not None:
            model_module_name = 'model_%s' % network_type
            model_class_name = underline_to_camel(model_module_name)
            try:
                src_module = __import__('src.models.%s.%s' % (network_class, model_module_name))
                model_class = eval('src_module.models.%s.%s.%s' % (network_class, model_module_name, model_class_name))
                model_params = model_class.get_default_model_parameters()
                model_params.add_hparam('model_class', model_class)  # add model class
            except ImportError:
                print('Fatal Error: no model module: \"src.models.%s.%s\"' % (network_class, model_module_name))
            except AttributeError:
                print('Fatal Error: probably (1) no model class named as %s.%s, '
                      'or (2) the class no \"get_default_model_parameters()\"' % (network_class, model_module_name))
        return model_params

    # ============== Utils =============
    def register_hparams(self, hparams, name):
        assert isinstance(hparams, HParams)
        assert isinstance(name, str)
        assert name not in self.hparam_name_list

        self.hparam_name_list.append(name)
        setattr(self, name, hparams)

    @property
    def all_params(self):
        all_params = HParams()
        for hparam_name in reversed(self.hparam_name_list):
            cur_params = getattr(self, hparam_name)
            all_params = merge_params(all_params, cur_params)
        return all_params

    def __getitem__(self, item):
        assert isinstance(item, str)

        for hparam_name in reversed(self.hparam_name_list):
            try:
                return getattr(getattr(self, hparam_name), item)
            except AttributeError:
                pass
        raise AttributeError('no item named as \'%s\'' % item)
