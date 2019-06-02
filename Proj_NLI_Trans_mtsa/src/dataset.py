
from tqdm import tqdm
import random
import math
import re
import os
import json
import nltk

import logging as log

from src.utils.bpe_enc import BpeTextEncoder


class Dataset(object):
    """
    Features:
        1. using BPE subword
        2.
    """
    START_TOKEN = '__start__'
    DELIMITER_TOKEN = '__delimiter__'
    CLASSIFY_TOKEN = '__classify__'

    def __init__(self, data_file_path, data_type, bpe_enc=None, bpe_path=None):

        self.data_type = data_type
        log.info('building data set object for %s' % data_type)
        assert data_type in ['train', 'dev', 'test']
        # check
        if data_type in ['dev', 'test']:
            assert bpe_enc is not None
        self.bpe_enc = bpe_enc or BpeTextEncoder(
            os.path.join(bpe_path, 'encoder_bpe_40000.json'),
            os.path.join(bpe_path, 'vocab_40000.bpe'),
            add_padding=True,
            special_tokens=(Dataset.START_TOKEN, Dataset.DELIMITER_TOKEN, Dataset.CLASSIFY_TOKEN)
        )

        raw_data = self._load_snli_data(data_file_path)
        self.nn_data = self._process_data(raw_data)

    @property
    def sample_num(self):
        return len(self.nn_data)

    @property
    def no_label(self):
        return 'gold_label' not in self.nn_data[0]

    def filter_example(self, max_sent_len):
        for example in self.nn_data:
            example['sentence1_token_digit'] = example['sentence1_token_digit'][:max_sent_len]
            example['sentence2_token_digit'] = example['sentence2_token_digit'][:max_sent_len]

    def sort_example(self):
        self.nn_data = list(
            sorted(
                self.nn_data,
                key=lambda ex: max(len(ex['sentence1_token_digit']), len(ex['sentence2_token_digit']))
            )
        )

    # get batch
    def generate_batch_iter(self, batch_size, max_step=None):
        if max_step is not None:
            def data_queue(data, _batch_size):
                assert len(data) >= batch_size
                random.shuffle(data)
                data_ptr = 0
                dataRound = 0
                idx_b = 0
                step = 0
                while True:
                    if data_ptr + _batch_size <= len(data):
                        yield data[data_ptr:data_ptr + _batch_size], dataRound, idx_b
                        data_ptr += _batch_size
                        idx_b += 1
                        step += 1
                    elif data_ptr + _batch_size > len(data):
                        offset = data_ptr + _batch_size - len(data)
                        out = data[data_ptr:]
                        random.shuffle(data)
                        out += data[:offset]
                        data_ptr = offset
                        dataRound += 1
                        yield out, dataRound, 0
                        idx_b = 1
                        step += 1
                    if step >= max_step:
                        break
            batch_num = math.ceil(len(self.nn_data) / batch_size)
            for sample_batch, data_round, idx_b in data_queue(self.nn_data, batch_size):
                yield sample_batch, batch_num, data_round, idx_b
        else:
            batch_num = math.ceil(len(self.nn_data) / batch_size)
            idx_b = 0
            sample_batch = []
            for sample in self.nn_data:
                sample_batch.append(sample)
                if len(sample_batch) == batch_size:
                    yield sample_batch, batch_num, 0, idx_b
                    idx_b += 1
                    sample_batch = []
            if len(sample_batch) > 0:
                yield sample_batch, batch_num, 0, idx_b

    def _process_data(self, raw_data):
        log.info('processing raw data for %s' % self.data_type)
        new_data = []
        for example in tqdm(raw_data):
            if 'gold_label' in example and example['gold_label'] == '-':
                continue

            example['sentence1_token'] = [Dataset.START_TOKEN] + self.bpe_enc.encode_sent(example['sentence1'], None) + \
                                         [Dataset.CLASSIFY_TOKEN]
            example['sentence2_token'] = [Dataset.START_TOKEN] + self.bpe_enc.encode_sent(example['sentence2'], None) + \
                                         [Dataset.CLASSIFY_TOKEN]

            # digitize
            example['sentence1_token_digit'] = [self.bpe_enc.get_idx_from_token(token)
                                                for token in example['sentence1_token']]
            example['sentence2_token_digit'] = [self.bpe_enc.get_idx_from_token(token)
                                                for token in example['sentence2_token']]

            # delete useless attrs
            useless_keys = [
                'annotator_labels', 'captionID', 'sentence1_binary_parse', 'sentence1_parse',
                'sentence2_binary_parse', 'sentence2_parse', 'genre', 'promptID']
            for useless_key in useless_keys:
                if useless_key in example:
                    example.pop(useless_key)

            # delete 'hidden' gold_label
            if example['gold_label'] == 'hidden':
                example.pop('gold_label')

            new_data.append(example)
        return new_data

    def _load_snli_data(self, data_path):
        log.info('load file for %s' % self.data_type)
        dataset = []
        with open(data_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line)
                dataset.append(json_obj)
        return dataset

