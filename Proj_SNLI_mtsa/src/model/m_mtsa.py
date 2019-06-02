from configs import cfg
from src.utils.record_log import _logger
import tensorflow as tf

from src.model.model_template import ModelTemplate
from src.nn_utils.nn import linear
from src.nn_utils.nn import generate_embedding_mat
from src.nn_utils.nn import highway_net
from src.nn_utils.self_attn import multi_dimensional_attention
from src.nn_utils.mtsa import multi_mask_tensorized_self_attn


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
        bs, sl1, sl2 = self.bs, self.sl1, self.sl2

        with tf.variable_scope('emb'):
            token_emb_mat = generate_embedding_mat(tds, tel, init_mat=self.token_emb_mat,
                                                   extra_mat=self.glove_emb_mat, extra_trainable=self.finetune_emb,
                                                   scope='gene_token_emb_mat')
            s1_emb = tf.nn.embedding_lookup(token_emb_mat, self.sent1_token)  # bs,sl1,tel
            s2_emb = tf.nn.embedding_lookup(token_emb_mat, self.sent2_token)  # bs,sl2,tel
            self.tensor_dict['s1_emb'] = s1_emb
            self.tensor_dict['s2_emb'] = s2_emb

        with tf.variable_scope('context_fusion'):
            s1_seq_rep = multi_mask_tensorized_self_attn(
                s1_emb, self.sent1_token_mask,
                hn=2*hn, head_num=2, is_train=self.is_train,
                attn_keep_prob=1., dense_keep_prob=cfg.dropout, wd=cfg.wd,
                use_direction=True, attn_self=False, use_fusion_gate=True, final_mask_ft=None,
                dot_activation_name='sigmoid', use_input_for_attn=False, add_layer_for_multi=True,
                activation_func_name='elu', apply_act_for_v=True, input_hn=None, output_hn=None,
                accelerate=False, merge_var=False,
                scope='multi_mask_tensorized_self_attn'
            )

            tf.get_variable_scope().reuse_variables()

            s2_seq_rep = multi_mask_tensorized_self_attn(
                s2_emb, self.sent2_token_mask,
                hn=2*hn, head_num=2, is_train=self.is_train,
                attn_keep_prob=1., dense_keep_prob=cfg.dropout, wd=cfg.wd,
                use_direction=True, attn_self=False, use_fusion_gate=True, final_mask_ft=None,
                dot_activation_name='sigmoid', use_input_for_attn=False, add_layer_for_multi=True,
                activation_func_name='elu', apply_act_for_v=True, input_hn=None, output_hn=None,
                accelerate=False, merge_var=False,
                scope='multi_mask_tensorized_self_attn'
            )

        with tf.variable_scope('compression'):
            s1_rep = multi_dimensional_attention(
                s1_seq_rep, self.sent1_token_mask, 's2t_attn', cfg.dropout, self.is_train, cfg.wd, 'elu'
            )

            tf.get_variable_scope().reuse_variables()

            s2_rep = multi_dimensional_attention(
                s2_seq_rep, self.sent2_token_mask, 's2t_attn', cfg.dropout, self.is_train, cfg.wd, 'elu'
            )

        with tf.variable_scope('output'):
            out_rep = tf.concat([s1_rep, s2_rep, s1_rep - s2_rep, s1_rep * s2_rep], -1)
            pre_output = tf.nn.elu(
                linear([out_rep], hn, True, 0., scope= 'pre_output', squeeze=False,
                       wd=cfg.wd, input_keep_prob=cfg.dropout,is_train=self.is_train))
            pre_output1 = highway_net(
                pre_output, hn, True, 0., 'pre_output1', 'elu', False, cfg.wd, cfg.dropout, self.is_train)
            logits = linear([pre_output1], self.output_class, True, 0., scope= 'logits', squeeze=False,
                            wd=cfg.wd, input_keep_prob=cfg.dropout,is_train=self.is_train)
            self.tensor_dict[logits] = logits
        return logits # logits
