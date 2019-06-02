"""
Author: Anonymity
Email: anonymity@email.com

Semantic Role Labeling with Multi-mask Tensorized Self-Attention (MTSA)

place this file to folder "Tagger/models" to replace original one.
!!!Please initialize trainable variables using "tf.glorot_normal_initializer()" and follow the instructions
    provided in README.md
"""


import ops
import copy
import tensorflow as tf

from ops.layers import layer_norm, linear
from ops.attention import multihead_attention
from ops.my_attn import split_head, combine_head, new_exp_mask, new_mask
import math


# read
def deepatt_default_params():
    params = tf.contrib.training.HParams(
        feature_size=100,
        hidden_size=200,
        filter_size=800,
        num_hidden_layers=10,
        residual_dropout=0.2,
        attention_dropout=0.1,
        relu_dropout=0.1,
        label_smoothing=0.1,
        fix_embedding=False,
        multiply_embedding_mode="sqrt_depth",
        num_heads=8,  # for multi_head
    )

    return params


def _residual_fn(x, y, params):
    if params.residual_dropout > 0.0:
        y = tf.nn.dropout(y, 1.0 - params.residual_dropout)

    return layer_norm(x + y)


def _ffn_layer(inputs, hidden_size, output_size, keep_prob=None,
               data_format="NHWC", dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = linear(inputs, hidden_size, True, data_format=data_format)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output = linear(hidden, output_size, True, data_format=data_format)

        return output


def encoder(encoder_input, mask, params, dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="encoder", dtype=dtype,
                           values=[encoder_input, mask]):
        x = encoder_input
        seq_len = tf.to_int32(tf.reduce_sum(mask, -1))
        attn_bias = ops.attention.attention_bias(mask, "masking")

        # multiple masks generation for multi-head
        rep_mask = tf.cast(mask, tf.bool)
        bs, sl = tf.shape(rep_mask)[0], tf.shape(rep_mask)[1]
        rep_mask_epd1 = tf.expand_dims(rep_mask, 1)  # bs,1,sl
        rep_mask_epd2 = tf.expand_dims(rep_mask, 2)  # bs,sl,1
        rep_mask_mat = tf.logical_and(rep_mask_epd1, rep_mask_epd2)  # bs,sl,sl

        sl_indices = tf.range(sl, dtype=tf.int32)
        sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)
        fw_mask = tf.greater(sl_row, sl_col)  # sl,sl
        bw_mask = tf.less(sl_row, sl_col)  # sl,sl
        direct_mask = tf.stack([fw_mask, bw_mask], 0)  # 2,sl,sl
        direct_mask = tf.reshape(  # num,sl,sl
            tf.tile(tf.expand_dims(direct_mask, 1), [1, int(params.num_heads/2), 1, 1]),  # 2,4,sl,sl
            [params.num_heads, sl, sl]
        )
        final_mask = tf.logical_and(  # num,bs,sl,sl
            tf.expand_dims(rep_mask_mat, 0),
            tf.expand_dims(direct_mask, 1),
        )
        final_mask_ft = tf.cast(final_mask, tf.float32)

        for layer in xrange(params.num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("computation"):
                    y = _ffn_layer(
                        x,
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, params)

                with tf.variable_scope("self_attention"):
                    y = multi_mask_tensorized_self_attn(
                        x, rep_mask, final_mask_ft, params.hidden_size, params.num_heads,
                        1.0 - params.attention_dropout
                    )
                    x = _residual_fn(x, y, params)
        return x


def deepatt_model(features, mode, params):
    hparams = params
    params = copy.copy(hparams)

    # disable dropout in evaluation/inference mode
    if mode != tf.contrib.learn.ModeKeys.TRAIN:
        params.attention_dropout = 0.0
        params.residual_dropout = 0.0
        params.relu_dropout = 0.0

    vocab_size = len(params.vocabulary["inputs"])
    label_size = len(params.vocabulary["targets"])
    hidden_size = params.hidden_size
    feature_size = params.feature_size

    tok_seq = features["inputs"]
    pred_seq = features["preds"]
    mask = tf.to_float(tf.not_equal(tok_seq, 0))

    # shared embedding and softmax weights
    initializer = None

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        if not params.use_global_initializer:
            initializer = tf.random_normal_initializer(0.0,
                                                       feature_size ** -0.5)

    weights = tf.get_variable("weights", [2, feature_size],
                              initializer=initializer)

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        if params.embedding is not None:
            initializer = lambda shape, dtype, partition_info: params.embedding
    else:
        initializer = None

    embedding = tf.get_variable("embedding", [vocab_size, feature_size],
                                initializer=initializer,
                                trainable=not params.fix_embedding)
    bias = tf.get_variable("bias", [hidden_size])

    # id => embedding
    # src_seq: [batch, max_src_length]
    # tgt_seq: [batch, max_tgt_length]
    inputs = tf.gather(embedding, tok_seq)

    if mode == tf.contrib.learn.ModeKeys.INFER:
        if features.get("mask") is not None:
            keep_mask = features["mask"][:, :, None]
            unk_emb = features["embedding"]
            inputs = inputs * keep_mask + (1.0 - keep_mask) * unk_emb

    preds = tf.gather(weights, pred_seq)
    inputs = tf.concat([inputs, preds], -1)

    if params.multiply_embedding_mode == "sqrt_depth":
        inputs = inputs * (hidden_size ** 0.5)

    inputs = inputs * tf.expand_dims(mask, -1)

    # preparing encoder & decoder input
    encoder_input = tf.nn.bias_add(inputs, bias)

    # !!! delete positional embedding in multi-head self-attention
    """
        if params.pos == "timing":
        encoder_input = ops.attention.add_timing_signal(encoder_input)
    elif params.pos == "embedding":
        initializer = tf.random_normal_initializer(0.0, hidden_size ** -0.5)
        embedding = tf.get_variable("position_embedding", [1000, hidden_size],
                                    initializer=initializer)
        indices = tf.range(tf.shape(features["inputs"])[1])[None, :]
        pos_emb = tf.gather(embedding, indices)
        pos_emb = tf.tile(pos_emb, [tf.shape(features["inputs"])[0], 1, 1])
        encoder_input = encoder_input + pos_emb

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)
    """

    encoder_output = encoder(encoder_input, mask, params)

    initializer = None

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        if not params.use_global_initializer:
            initializer = tf.random_normal_initializer(0.0,
                                                       hidden_size ** -0.5)

    with tf.variable_scope("prediction", initializer=initializer):
        logits = linear(encoder_output, label_size, True, scope="logits")

    if mode == tf.contrib.learn.ModeKeys.INFER:
        outputs = tf.to_int32(tf.argmax(logits, axis=-1))
        return outputs, tf.nn.softmax(logits)

    labels = features["targets"]
    targets = features["targets"]
    logits = tf.reshape(logits, [-1, label_size])
    labels = tf.reshape(labels, [-1])

    # label smoothing
    ce = ops.layers.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        label_smoothing=params.label_smoothing,
        normalize=True
    )

    ce = tf.reshape(ce, tf.shape(targets))
    cost = tf.reduce_sum(ce * mask) / tf.reduce_sum(mask)

    # greedy decoding
    if mode == tf.contrib.learn.ModeKeys.EVAL:
        outputs = tf.to_int32(tf.argmax(logits, axis=-1))
        return cost, tf.reshape(outputs, tf.shape(targets))

    return cost


def multi_mask_tensorized_self_attn(
        rep_tensor, rep_mask, final_mask_ft, hn, head_num, keep_prob=None, scope=None):
    data_format = "NHWC"
    assert hn % head_num == 0, "hn (%d) must be divisible by the number of " \
                         "attention heads (%d)." % (hn, head_num)
    head_dim = int(hn / head_num)

    bs, sl = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1]
    with tf.variable_scope(scope or 'proposed_self_attention'):
        combined = linear(rep_tensor, 3 * hn, True, True, data_format=data_format,
                          scope="qkv_transform")
        q, k, v = tf.split(combined, 3, 2)  # bs,sl,hn

        q = split_head(q, head_num)
        k = split_head(k, head_num)
        v = split_head(v, head_num)  # num,bs,sl,dim

        with tf.name_scope("dot_product_attention"):
            dot_logits = tf.matmul(q, k, transpose_b=True) * (head_dim ** -0.5)  # num,bs,sl,sl
            e_dot_logits = tf.exp(new_exp_mask(dot_logits, final_mask_ft))  # num,bs,sl,sl

        with tf.variable_scope("s2t_multi_dim_attention"):
            multi_logits_before = linear(
                rep_tensor, hn, True, True, data_format=data_format, scope="multi_logits_before")
            multi_logits = split_head(multi_logits_before, head_num)  # num,bs,sl,dim
            e_multi_logits = tf.exp(new_exp_mask(  # mul,bs,sl,dim
                multi_logits, rep_mask, multi_head=True, high_dim=True))

        with tf.name_scope("hybrid_attn"):
            accum_z_deno = tf.matmul(e_dot_logits, e_multi_logits)  # num,bs,sl,dim
            accum_z_deno = tf.where(  # in case of nan
                tf.greater(accum_z_deno, tf.zeros_like(accum_z_deno)),
                accum_z_deno,
                tf.ones_like(accum_z_deno)
            )
            if keep_prob is not None and keep_prob < 1.0:
                real_keep_prob = keep_prob
                e_multi_logits = tf.nn.dropout(e_multi_logits, real_keep_prob)
                e_dot_logits = tf.nn.dropout(e_dot_logits, real_keep_prob)

            rep_mul_score = new_mask(v, rep_mask, multi_head=True, high_dim=True) * e_multi_logits
            accum_rep_mul_score = tf.matmul(e_dot_logits, rep_mul_score)
            attn_res = accum_rep_mul_score / accum_z_deno

        with tf.variable_scope("output"):
            attn_output = combine_head(attn_res)  # bs,sl,hn
            final_out = linear(attn_output, hn, True, data_format=data_format,
                               scope="output_transform")

        final_out = new_mask(final_out, rep_mask, high_dim=True) # bs,sl,hn
        return final_out

