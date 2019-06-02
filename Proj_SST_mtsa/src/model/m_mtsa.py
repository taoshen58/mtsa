from configs import cfg
from src.utils.record_log import _logger
import tensorflow as tf

from src.model.template import ModelTemplate
from src.nn_utils.nn import linear
from src.nn_utils.nn import generate_embedding_mat
from src.nn_utils.mtsa import multi_mask_tensorized_self_attn, multi_dim_souce2token_self_attn


class ModelMTSA(ModelTemplate):
    def __init__(self, token_emb_mat, glove_emb_mat, tds, cds, tl, scope):
        super(ModelMTSA, self).__init__(token_emb_mat, glove_emb_mat, tds, cds, tl, scope)
        self.update_tensor_add_ema_and_opt()

    def build_network(self):
        _logger.add()
        _logger.add('building %s neural network structure...' % cfg.network_type)
        tds, cds = self.tds, self.cds
        tl = self.tl
        tel, cel, cos, ocd, fh = self.tel, self.cel, self.cos, self.ocd, self.fh
        hn = self.hn
        bs, sl, ol, mc = self.bs, self.sl, self.ol, self.mc

        with tf.variable_scope('emb'):
            token_emb_mat = generate_embedding_mat(tds, tel, init_mat=self.token_emb_mat,
                                                   extra_mat=self.glove_emb_mat, extra_trainable=self.finetune_emb,
                                                   scope='gene_token_emb_mat')
            emb = tf.nn.embedding_lookup(token_emb_mat, self.token_seq)  # bs,sl,tel
            self.tensor_dict['emb'] = emb

        with tf.variable_scope('sent_encoding'):
            act_name = 'relu'
            seq_rep = multi_mask_tensorized_self_attn(
                emb, self.token_mask,
                hn=2 * hn, head_num=2, is_train=self.is_train,
                attn_keep_prob=1., dense_keep_prob=cfg.dropout, wd=cfg.wd,
                use_direction=True, attn_self=False, use_fusion_gate=True, final_mask_ft=None,
                dot_activation_name='sigmoid', use_input_for_attn=False, add_layer_for_multi=True,
                activation_func_name=act_name, apply_act_for_v=True, input_hn=None, output_hn=None,
                accelerate=False, merge_var=False,
                scope='proposed_model'
            )

            rep = multi_dim_souce2token_self_attn(
                seq_rep, self.token_mask, 's2t_self_attn', cfg.dropout, self.is_train, cfg.wd, act_name
            )

        with tf.variable_scope('output'):
            pre_logits = tf.nn.relu(linear([rep], hn, True, scope='pre_logits_linear',
                                          wd=cfg.wd, input_keep_prob=cfg.dropout,
                                          is_train=self.is_train))  # bs, hn
            logits = linear([pre_logits], self.output_class, False, scope='get_output',
                            wd=cfg.wd, input_keep_prob=cfg.dropout, is_train=self.is_train) # bs, 5
        _logger.done()
        return logits


