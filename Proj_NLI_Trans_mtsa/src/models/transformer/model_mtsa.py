import tensorflow as tf
import numpy as np

import math

from src.utils.hparams import HParams, merge_params
from src.models.transformer.model_template import ModelTemplate
from src.models.transformer.modules import add_pos_emb_idx, get_emb_matrix, multi_head_block, \
    get_transformer_clf_features, language_model, snli_logits, openai_adam, schedules_name2fn
from src.nn_utils.mtsa import stacking_mtsa, multi_dim_souce2token_self_attn, multi_mask_tensorized_self_attn, \
    bn_dense_layer


class ModelMtsa(ModelTemplate):
    @staticmethod
    def get_default_model_parameters():
        return merge_params(
            ModelTemplate.get_default_model_parameters(),
            HParams(
                token_rep_highway="none", # [none|mlp,gate,add]
                seq_rep_highway="none", # [none|mlp,gate,add]
            )
        )

    @staticmethod
    def get_identity_param_list():
        return ModelTemplate.get_identity_param_list() + [
            "token_rep_highway", "seq_rep_highway"]

    def __init__(self, cfg, n_vocab, n_special, clf_token, scope):
        super(ModelMtsa, self).__init__(cfg, n_vocab, n_special, clf_token, scope)
        self._training_setup()

    def _build_network(self):
        low_out = self._build_low_level_network()
        seq_h_rsp, seq_h, lm_losses = low_out['seq_h_rsp'], low_out['seq_h'], low_out['lm_losses']

        # >>>> merge emb_rsp and seq_h_rsp
        if self.cfg['token_rep_highway'] == 'none':
            pass
        else:
            emb_rsp = low_out['emb_rsp']
            if self.cfg['token_rep_highway'] == 'mlp':
                seq_h_rsp = bn_dense_layer(
                    tf.concat([emb_rsp, seq_h_rsp], axis=-1),
                    self.cfg['embd_dim'], True, 0., 'merge_emb_token_h',
                    self.cfg['clf_afn'], False, 0., self.cfg['clf_dropout'], self.is_train
                )
            elif self.cfg['token_rep_highway'] == 'gate':
                merge_gate_token = bn_dense_layer(
                    tf.concat([emb_rsp, seq_h_rsp], axis=-1),
                    self.cfg['embd_dim'], True, 0., 'merge_emb_token_h',
                    'sigmoid', False, 0., self.cfg['clf_dropout'], self.is_train
                )
                seq_h_rsp = merge_gate_token * emb_rsp + (1 - merge_gate_token) * seq_h_rsp
            elif self.cfg['token_rep_highway'] == 'add':
                seq_h_rsp = emb_rsp + seq_h_rsp
            else:
                raise AttributeError

        #  >>>> main MTSA model
        seq_h_rsp = multi_mask_tensorized_self_attn( # [bs*2,sl,dim]
            seq_h_rsp, self.sent_token_mask_rsp, self.cfg['embd_dim'], self.cfg['n_head'], self.is_train,
            self.cfg['clf_dropout'], self.cfg['clf_dropout'], 0.,
            use_direction=True, attn_self=False, use_fusion_gate=True, final_mask_ft=None,
            dot_activation_name='sigmoid', use_input_for_attn=False, add_layer_for_multi=True,
            activation_func_name=self.cfg['clf_afn'], apply_act_for_v=True, input_hn=None, output_hn=None,
            accelerate=False, merge_var=False,
            scope='multi_mask_tensorized_self_attn',
        )

        # >>>> source2token self-attention
        reps_rsp = multi_dim_souce2token_self_attn(  # [bs*2,dim]
            seq_h_rsp, self.sent_token_mask_rsp, 'multi_dim_souce2token_self_attn', self.cfg['clf_dropout'],
            self.is_train, 0., activation=self.cfg['clf_afn']
        )

        # >>>> merge low_out['seq_h_rsp'] and reps_rsp
        if self.cfg['seq_rep_highway'] == 'none':
            pass
        else:
            reps_rsp_lm = get_transformer_clf_features(seq_h_rsp, self.sent_token_rsp, self.clf_token)
            if self.cfg['seq_rep_highway'] == 'mlp':
                reps_rsp = tf.squeeze(bn_dense_layer(
                    tf.expand_dims(tf.concat([reps_rsp, reps_rsp_lm], axis=-1), axis=1),
                    self.cfg['embd_dim'], True, 0., 'merge_emb_seq_h',
                    self.cfg['clf_afn'], False, 0., self.cfg['clf_dropout'], self.is_train
                ), axis=[1])
            elif self.cfg['seq_rep_highway'] == 'gate':
                merge_gate_seq = tf.squeeze(bn_dense_layer(
                    tf.expand_dims(tf.concat([reps_rsp, reps_rsp_lm], axis=-1), axis=1),
                    self.cfg['embd_dim'], True, 0., 'merge_emb_seq_h',
                    'sigmoid', False, 0., self.cfg['clf_dropout'], self.is_train
                ), axis=[1])
                reps_rsp = merge_gate_seq * reps_rsp + (1 - merge_gate_seq) * reps_rsp_lm
            elif self.cfg['seq_rep_highway'] == 'add':
                reps_rsp = reps_rsp + reps_rsp_lm
            else:
                raise AttributeError



        # for the task
        # reps_rsp = get_transformer_clf_features(seq_h_rsp, self.sent_token_rsp, self.clf_token)  #
        reps = tf.reshape(reps_rsp, [self.bs, 2, self.cfg['embd_dim']])
        logits = snli_logits(  # [bs,3]
            reps[:, 0], reps[:, 1], self.cfg['afn'], self.cfg['embd_dim'],
            self.is_train, self.cfg['clf_dropout'], self.cfg['highway'])
        task_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.gold_label,
            logits=logits
        )
        return logits, task_losses, lm_losses
