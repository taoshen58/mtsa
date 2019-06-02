import tensorflow as tf
import numpy as np

import math

from src.utils.hparams import HParams, merge_params
from src.models.transformer.model_template import ModelTemplate
from src.models.transformer.modules import add_pos_emb_idx, get_emb_matrix, multi_head_block, \
    get_transformer_clf_features, language_model, snli_logits, openai_adam, schedules_name2fn
from src.nn_utils.mtsa import multi_dim_souce2token_self_attn

class ModelTransformer(ModelTemplate):
    @staticmethod
    def get_default_model_parameters():
        return merge_params(
            ModelTemplate.get_default_model_parameters(),
            HParams(
                use_mtsa=False,
                use_direction=False,
                output_method="clf",
                global_afn='exp',
                attn_self=True,
            )
        )

    @staticmethod
    def get_identity_param_list():
        return ModelTemplate.get_identity_param_list() + [
            'use_mtsa', 'use_direction', 'output_method', 'global_afn', 'attn_self']

    def __init__(self, cfg, n_vocab, n_special, clf_token, scope):
        super(ModelTransformer, self).__init__(cfg, n_vocab, n_special, clf_token, scope)
        self._training_setup()

    def _build_network(self):
        low_out = self._build_low_level_network()
        seq_h_rsp, seq_h, lm_losses = low_out['seq_h_rsp'], low_out['seq_h'], low_out['lm_losses']

        for idx_h_l in range(1):
            seq_h_rsp = multi_head_block(
                seq_h_rsp, 'top_h%d' % idx_h_l, self.is_train, True, self.cfg['n_head'], self.cfg['clf_afn'],
                self.cfg['clf_dropout'], self.cfg['clf_dropout'], reuse=False,
                use_global=self.cfg['use_mtsa'], use_direction=self.cfg['use_direction'],
                x_mask=self.sent_token_mask_rsp, global_afn=self.cfg['global_afn'], attn_self=self.cfg['attn_self'],
            )

        if self.cfg['output_method'] == 's2t':
            reps_rsp = multi_dim_souce2token_self_attn(  # [bs*2,dim]
                seq_h_rsp, self.sent_token_mask_rsp, 'multi_dim_souce2token_self_attn', self.cfg['clf_dropout'],
                self.is_train, 0., activation=self.cfg['clf_afn']
            )
        elif self.cfg['output_method'] == 'clf':
            reps_rsp = get_transformer_clf_features(seq_h_rsp, self.sent_token_rsp, self.clf_token)
        else:
            raise AttributeError
        # for the task
        # reps_rsp = get_transformer_clf_features(seq_h_rsp, self.sent_token_rsp, self.clf_token)  #
        reps = tf.reshape(reps_rsp, [self.bs, 2, self.cfg['embd_dim']])
        logits = snli_logits(  # [bs,3]
            reps[:, 0], reps[:, 1], self.cfg['clf_afn'], self.cfg['embd_dim'], self.is_train, self.cfg['clf_dropout'])
        task_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.gold_label,
            logits=logits
        )
        return logits, task_losses, lm_losses
