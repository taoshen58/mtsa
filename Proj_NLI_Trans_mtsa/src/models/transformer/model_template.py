from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
import logging as log
import os
import json
import re

from functools import partial

from src.utils.hparams import HParams, merge_params
from src.models.transformer.modules import add_pos_emb_idx, get_emb_matrix, multi_head_block, \
    get_transformer_clf_features, language_model, snli_logits, openai_adam, schedules_name2fn


class ModelTemplate(metaclass=ABCMeta):
    @staticmethod
    def get_default_model_parameters():
        return HParams(
            embd_dim=768,
            n_hidden=768,
            n_ctx=512,
            embd_dropout=0.9,
            resid_dropout=0.9,
            attn_dropout=0.9,
            clf_dropout=0.9,
            n_layer=12,
            # n_transfer=12,
            n_head=12,
            afn='gelu',
            clf_afn='elu',
            lm_coef=0.3,
            lr_schd='warmup_linear',
            lr_warmup=0.002,
            highway=False,
            use_pe=True,
        )

    @staticmethod
    def get_identity_param_list():
        return ['clf_afn', 'lm_coef', 'clf_dropout', 'resid_dropout', 'highway', 'n_layer',
                'lr_schd', 'lr_warmup', 'use_pe']

    def __init__(self, cfg, n_vocab, n_special, clf_token, scope):
        self.cfg = cfg
        self.n_vocab = n_vocab
        self.n_special = n_special
        self.clf_token = clf_token
        self.scope = scope

        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        # ---- place holder -----
        # self.sent1_token = tf.placeholder(tf.int32, [None, None], name='sent1_token')
        # self.sent2_token = tf.placeholder(tf.int32, [None, None], name='sent2_token')
        self.sent_token = tf.placeholder(tf.int32, [None, 2, None], name='sent_token')
        self.gold_label = tf.placeholder(tf.int32, [None], name='gold_label')
        self.is_train = tf.placeholder(tf.bool, [], name='is_train')

        # ----------- parameters -------------
        self.hn = self.cfg['n_hidden']

        self.bs = tf.shape(self.sent_token)[0]
        self.sl = tf.shape(self.sent_token)[2]

        # ------------ other ---------
        self.sent_token_mask = tf.cast(self.sent_token, tf.bool)
        self.sent_token_len = tf.reduce_sum(tf.cast(self.sent_token_mask, tf.int32), -1)

        self.sent_token_rsp = tf.reshape(self.sent_token, [self.bs*2, self.sl])
        self.sent_token_mask_rsp = tf.reshape(self.sent_token_mask, [self.bs*2, self.sl])
        self.sent_token_len_rsp = tf.reshape(self.sent_token_len, [self.bs*2])
        #
        # self.sent1_token = self.sent_token[:, 0]
        # self.sent1_token_mask = self.sent_token_mask[:, 0]
        # self.sent1_token_len = self.sent_token_len[:, 0]
        #
        # self.sent2_token = self.sent_token[:, 1]
        # self.sent2_token_mask = self.sent_token_mask[:, 1]
        # self.sent2_token_len = self.sent_token_len[:, 1]

        self.tensor_dict = {}

        #
        self.logits = None
        self.task_losses = None
        self.lm_losses = None
        self.loss = None
        self.task_loss = None
        self.lm_loss = None
        self.accuracy = None
        self.accuracy_mean = None
        self.train_op = None
        self.trainable_vars = None
        self.summary = None

        # def step(self, sess, batch_samples, get_summary=False):
    def step(self, sess, batch_data, summary_period=None):
        global_step_val = sess.run(self.global_step) + 1
        feed_dict = self.get_feed_dict(batch_data, True)

        if isinstance(summary_period, int) and global_step_val % summary_period == 0:
            _, task_loss, lm_loss, loss, accu, summary = sess.run(
                [self.train_op, self.task_loss, self.lm_loss, self.loss, self.accuracy_mean, self.summary],
                feed_dict=feed_dict
            )
        else:
            _, task_loss, lm_loss, loss, accu = sess.run(
                [self.train_op, self.task_loss, self.lm_loss, self.loss, self.accuracy_mean],
                feed_dict=feed_dict)
            summary = None
        return {'task_loss':task_loss, 'lm_loss': lm_loss,'loss': loss, 'accuracy': accu, 'summary': summary}

    def _build_network(self):
        # token_w_pos = add_pos_emb_idx(self.sent_token, self.n_vocab)  # [bs,2,sl,2]
        #
        # we, we_vocab = get_emb_matrix(
        #     self.n_vocab, self.cfg['n_ctx'], self.cfg['embd_dim'], self.is_train, self.cfg['embd_dropout'])
        #
        # seq_emb_w_pos = tf.nn.embedding_lookup(we, token_w_pos)
        # seq_emb = tf.reduce_sum(seq_emb_w_pos, axis=-2)
        #
        # # stacking multi-head attention
        # seq_h_rsp = tf.reshape(seq_emb, [self.bs*2, self.sl, self.cfg['embd_dim']])  # bs*2,sl,dim
        # for idx_l in range(self.cfg['n_layer']):
        #     seq_h_rsp = multi_head_block(
        #         seq_h_rsp, 'h%d'%idx_l, self.is_train, True, self.cfg['n_head'], self.cfg['afn'],
        #         self.cfg['resid_dropout'], self.cfg['attn_dropout'], reuse=False
        #     )
        #
        # # for language model
        # lm_losses = language_model(
        #     seq_h_rsp, self.sent_token_rsp, self.sent_token_mask_rsp, we_vocab
        # )
        low_out = self._build_low_level_network()
        seq_h_rsp, seq_h, lm_losses = low_out['seq_h_rsp'], low_out['seq_h'], low_out['lm_losses']

        # for the task
        reps_rsp = get_transformer_clf_features(seq_h_rsp, self.sent_token_rsp, self.clf_token)  #
        reps = tf.reshape(reps_rsp, [self.bs, 2, self.cfg['embd_dim']])
        logits = snli_logits(  # [bs,3]
            reps[:, 0], reps[:, 1], self.cfg['afn'], self.cfg['embd_dim'],
            self.is_train, self.cfg['clf_dropout'], self.cfg['highway'])
        task_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.gold_label,
            logits=logits
        )

        return logits, task_losses, lm_losses

    def _build_low_level_network(self, n_layers_for_lm=None, use_global=False):
        n_layers_for_lm = n_layers_for_lm or self.cfg['n_layer']

        if self.cfg['use_pe']:
            token_w_pos = add_pos_emb_idx(self.sent_token, self.n_vocab)  # [bs,2,sl,2]
        else:
            token_w_pos = self.sent_token  # [bs,2,sl]

        we, we_vocab = get_emb_matrix(
            self.n_vocab, self.cfg['n_ctx'], self.cfg['embd_dim'], self.is_train, self.cfg['embd_dropout'],
            scope="emb_openai_trans"
        )

        seq_emb_w_pos = tf.nn.embedding_lookup(we, token_w_pos)
        if self.cfg['use_pe']:
            seq_emb = tf.reduce_sum(seq_emb_w_pos, axis=-2)
        else:
            seq_emb = seq_emb_w_pos

        # stacking multi-head attention
        emb_rsp = tf.reshape(seq_emb, [self.bs * 2, self.sl, self.cfg['embd_dim']])  # bs*2,sl,dim

        seq_h_rsp = emb_rsp
        for idx_l in range(n_layers_for_lm):
            seq_h_rsp = multi_head_block(
                seq_h_rsp, 'h%d' % idx_l, self.is_train, True, self.cfg['n_head'], self.cfg['afn'],
                self.cfg['resid_dropout'], self.cfg['attn_dropout'], reuse=False, use_global=use_global
            )
        seq_h = tf.reshape(seq_h_rsp, [self.bs, 2, self.sl, self.cfg['embd_dim']])

        # for language model
        lm_losses = language_model(
            seq_h_rsp, self.sent_token_rsp, self.sent_token_mask_rsp, we_vocab
        )
        return {
            "seq_h_rsp": seq_h_rsp, "seq_h": seq_h, "lm_losses": lm_losses,
            'emb_rsp': emb_rsp,
        }



    def _build_loss(self):
        # task loss
        task_loss = tf.reduce_mean(self.task_losses, name='task_loss')
        tf.summary.scalar(task_loss.op.name, task_loss)
        tf.add_to_collection('losses', task_loss)

        # language model loss
        if self.cfg['lm_coef'] > 0:
            lm_loss = tf.reduce_mean(self.lm_losses, name='lm_loss')
            tf.summary.scalar(lm_loss.op.name, lm_loss)
            tf.add_to_collection('losses', lm_loss * self.cfg['lm_coef'])
        else:
            lm_loss = tf.zeros([])
        # sum all loss
        loss = tf.add_n(tf.get_collection('losses', self.scope), name='loss')
        tf.summary.scalar(loss.op.name, loss)
        return loss, task_loss, lm_loss

    def _build_accuracy(self):
        correct = tf.equal(
            tf.cast(tf.argmax(self.logits, -1), tf.int32),
            self.gold_label
        )  # [bs]
        return tf.cast(correct, tf.float32), tf.reduce_mean(tf.cast(correct, tf.float32))

    def _training_setup(self):
        self.logits, self.task_losses, self.lm_losses = self._build_network()
        self.loss, self.task_loss, self.lm_loss = self._build_loss()
        self.accuracy, self.accuracy_mean = self._build_accuracy()

        # ======= ema =========== todo

        self.summary = tf.summary.merge_all()
        # ========== opt==========
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        self.trainable_vars = trainable_vars

        all_params_num = 0
        for elem in trainable_vars:
            # elem.name
            var_name = elem.name.split(':')[0]
            if var_name.endswith('we'): continue
            params_num = 1
            for l in elem.get_shape().as_list():
                params_num *= l
            all_params_num += params_num
        log.info('Trainable Parameters Number: %d (except emb part)' % all_params_num)

        grads = tf.gradients(self.loss, trainable_vars)

        if self.cfg['optimizer'] == 'openai_adam':
            self.train_op = openai_adam(
                trainable_vars,
                grads,
                self.cfg['lr'],
                partial(schedules_name2fn(self.cfg['lr_schd']), warmup=self.cfg['lr_warmup']),
                self.cfg['n_steps'], e=1e-8, l2=0.01, vector_l2=False, max_grad_norm=self.cfg['grad_norm'],
                global_step=self.global_step,
            )
        else:
            grads, _ = tf.clip_by_global_norm(grads, self.cfg['grad_norm'])
            if self.cfg['optimizer'] == 'adadelta':
                opt = tf.train.AdadeltaOptimizer(self.cfg['lr'])
                self.train_op = opt.apply_gradients(zip(grads, trainable_vars), self.global_step)
            else:
                raise(NotImplementedError, self.cfg['optimizer'])

    def get_trainable_vars(self, keys=tuple()):
        assert isinstance(keys, (tuple, list))
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

        if len(keys) == 0:
            return trainable_vars
        else:
            regex_pattern = ".*{}.*".format(".*".join(keys))
            new_trainable_vars = []
            for var in trainable_vars:
                if re.match(regex_pattern, var.op.name):
                    new_trainable_vars.append(var)
            return new_trainable_vars


    def load_openai_pretrained_transformer_model(self, sess, pretrain_dir):
        openai_vars = self.get_trainable_vars(keys=('openai_trans',))

        shapes = json.load(open(os.path.join(pretrain_dir, 'params_shapes.json')))
        offsets = np.cumsum([np.prod(shape) for shape in shapes])
        init_params = [np.load(os.path.join(pretrain_dir, 'params_{}.npy'.format(n))) for n in range(10)]
        init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
        init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
        init_params[0] = init_params[0][:self.cfg['n_ctx']]
        init_params[0] = np.concatenate(
            [
                init_params[1],
                (np.random.randn(self.n_special, self.cfg['embd_dim']) * 0.02).astype(np.float32),
                init_params[0]],
            0)
        del init_params[1]

        if self.cfg['n_layer'] == -1:
            n_transfer = 0
        else:
            n_transfer = 1 + self.cfg['n_layer'] * 12

        sess.run([p.assign(ip) for p, ip in zip(openai_vars[:n_transfer], init_params[:n_transfer])])

    def get_feed_dict(self, sample_batch, is_train):
        # max lens
        sl = 0
        for sample in sample_batch:
            sl = max(sl, len(sample['sentence1_token_digit']), len(sample['sentence2_token_digit']))

        # token
        sent1_token_b = []
        sent2_token_b = []
        for sample in sample_batch:
            sent1_token = np.zeros([sl], self.cfg['intX'])
            for idx_t, token in enumerate(sample['sentence1_token_digit']):
                sent1_token[idx_t] = token

            sent2_token = np.zeros([sl], self.cfg['intX'])

            for idx_t, token in enumerate(sample['sentence2_token_digit']):
                sent2_token[idx_t] = token
            sent1_token_b.append(sent1_token)
            sent2_token_b.append(sent2_token)
        sent1_token_b = np.stack(sent1_token_b, axis=0)  # bs,sl
        sent2_token_b = np.stack(sent2_token_b, axis=0)  # bs,sl

        sent_token = np.stack([sent1_token_b, sent2_token_b], axis=1)  # bs,2,sl

        feed_dict = {
            # self.sent1_token: sent1_token_b,
            # self.sent2_token: sent2_token_b,
            self.sent_token: sent_token,
            self.is_train: is_train,
        }

        # label
        if 'gold_label' in sample_batch[0]:
            gold_label_b = []
            for sample in sample_batch:
                gold_label_int = None
                if sample['gold_label'] == 'contradiction':
                    gold_label_int = 0
                elif sample['gold_label'] == 'neutral':
                    gold_label_int = 1
                elif sample['gold_label'] == 'entailment':
                    gold_label_int = 2
                assert gold_label_int is not None, sample['gold_label']
                gold_label_b.append(gold_label_int)
            gold_label_b = np.stack(gold_label_b).astype(self.cfg['intX'])
            feed_dict[self.gold_label] = gold_label_b


        return feed_dict


