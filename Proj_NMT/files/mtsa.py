import tensorflow as tf


from tensor2tensor.layers import common_layers, common_attention
from tensor2tensor.utils import expert_utils
import math


def dot_product_attention_mtsa(
        q,
        k,
        v,
        bias,
        dropout_rate=0.0,
        image_shapes=None,
        name=None,
        make_image_summary=True,
        save_weights_to=None,
        dropout_broadcast_dims=None,
        use_k_mtsa=True,
        afn_extra='none',
        afn_dot='exp',
        afn_multi='exp',
        bias_start=0.,
        bi_direction=False,
):
  """Dot-product attention.

  Args:
    q: Tensor with shape [..., length_q, depth_k].
    k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
      match with q.
    v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
      match with q.
    bias: bias Tensor (see attention_bias())
    dropout_rate: a float.
    image_shapes: optional tuple of integer scalars.
      see comments for attention_image_summary()
    name: an optional string
    make_image_summary: True if you want an image summary.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    dropout_broadcast_dims: an optional list of integers less than rank of q.
      Specifies in which dimensions to broadcast the dropout decisions.

  Returns:
    Tensor with shape [..., length_q, depth_v].
  """
  print("!!!!!dot_product_attention_mtsa!!!!!")
  with tf.variable_scope(
      name, default_name="dot_product_attention", values=[q, k, v]) as scope:
    # get dim
    dim_q = q.get_shape().as_list()[-1]
    dim_k = k.get_shape().as_list()[-1]
    dim_v = v.get_shape().as_list()[-1]
    # prepare
    multi_logits_scale_factor = 1./math.sqrt(dim_v) if afn_multi.startswith('scaled') else 1.
    afn_extra, afn_dot, afn_multi= afn_name2fn(afn_extra), afn_name2fn(afn_dot), afn_name2fn(afn_multi)
    # if bias is not None:
    #   inp_mask_1d = tf.to_float(tf.equal(bias, 0.))  # bs,1,1,vl
    #   inp_mask_1d = tf.transpose(inp_mask_1d, [0, 1, 3, 2])   # bs,1,vl,1
    # else:
    #   inp_mask_1d = None

    # token2token self attention
    dot_logits = tf.matmul(q, k, transpose_b=True)  # bs,hd,ql,vl
    if bias is not None:
      bias = common_layers.cast_like(bias, dot_logits)  # 1/bs,1,ql/1,vl
      dot_logits += bias
    e_dot_logits = afn_dot(dot_logits)  # bs,hd,ql,vl
    if bi_direction:
      head_num = v.get_shape().as_list()[1]
      ql, vl = tf.shape(q)[-2], tf.shape(v)[-2]
      assert head_num is not None
      assert head_num % 2 == 0
      ones_mat = tf.ones([ql, vl], tf.float32)
      mul_mask_fw = tf.matrix_band_part(ones_mat, -1, 0) #  Lower triangular part.
      mul_mask_bw = tf.matrix_band_part(ones_mat, 0, -1) #  Upper triangular part.
      mul_mask_fw_tile = tf.tile(tf.expand_dims(mul_mask_fw, 0), [head_num//2, 1, 1])
      mul_mask_bw_tile = tf.tile(tf.expand_dims(mul_mask_bw, 0), [head_num//2, 1, 1])
      mul_mask = tf.expand_dims(tf.concat([mul_mask_fw_tile, mul_mask_bw_tile], axis=0), axis=0)
      e_dot_logits *= mul_mask

    # source2token self-attention
    multi_logits = multi_head_dense_layer(
      k if use_k_mtsa else v, dim_v, True, bias_start if afn_extra is None else 0., 'multi_logits1')
    if afn_extra is not None:  # use one extra layer for multi-dim
      multi_logits = multi_head_dense_layer(afn_extra(multi_logits), dim_v, True, bias_start, 'multi_logits2')
    e_multi_logits = afn_multi(multi_logits * multi_logits_scale_factor) # bs,hd,vl,vd
    # if inp_mask_1d is not None:  # use mask for exp_logits
    #   e_multi_logits *= inp_mask_1d

    # mtsa
    accum_z_deno = tf.matmul(e_dot_logits, e_multi_logits)  # bs,hd,ql,vd
    accum_z_deno = tf.where(  # in case of NaN and Inf
      tf.greater(accum_z_deno, tf.zeros_like(accum_z_deno)),
      accum_z_deno,
      tf.ones_like(accum_z_deno)
    )

    # attention dropout
    e_dot_logits = common_layers.dropout_with_broadcast_dims(
      e_dot_logits, math.sqrt(1. - dropout_rate), broadcast_dims=dropout_broadcast_dims)
    e_multi_logits = common_layers.dropout_with_broadcast_dims(
      e_multi_logits, math.sqrt(1. - dropout_rate), broadcast_dims=dropout_broadcast_dims)
    rep_mul_score = v * e_multi_logits  # bs,hd,vl,vd
    accum_rep_mul_score = tf.matmul(e_dot_logits, rep_mul_score)  # bs,hd,ql,vd
    # calculate the final attention results
    attn_res = accum_rep_mul_score / accum_z_deno
    # if inp_mask_1d is not None:  # use mask for output
    #   attn_res *= inp_mask_1d

    # ============ for vis =======
    weights = e_dot_logits / (tf.reduce_sum(e_dot_logits, axis=-1, keepdims=True, name="attention_weights")+0.00001)
    if save_weights_to is not None:
      save_weights_to[scope.name] = weights
      save_weights_to[scope.name + "/logits"] = dot_logits
    if common_layers.should_generate_summaries() and make_image_summary:
      common_attention.attention_image_summary(weights, image_shapes)
    return attn_res


# ================= Utils =======================
def afn_name2fn(afn_name):
  if afn_name=='none':
    return None
  elif afn_name=='linear':
    return tf.identity
  elif afn_name=='relu':
    return tf.nn.relu
  elif afn_name == 'elu':
    return tf.nn.elu
  elif afn_name=='leaky_relu':
    return tf.nn.leaky_relu
  elif afn_name=='exp':
    return tf.exp
  elif afn_name=='sigmoid':
    return tf.sigmoid
  elif afn_name=='scaled_exp':
    return tf.exp
  elif afn_name=='scaled_sigmoid':
    return tf.sigmoid
  elif afn_name=='tanh':
    return tf.tanh
  elif afn_name=='ones':  # for debug
    return lambda inp: tf.ones_like(inp)
  else:
    raise AttributeError


def multi_head_dense_layer(
  input_tensor_trans, hn, bias, bias_start=0.0,
  scope=None, dup_num=1, merge_var=False
):  # [bs,hd,sl,dim]
  with tf.variable_scope(scope, default_name='multi_head_dense_layer'):
    input_tensor = tf.transpose(input_tensor_trans, [1, 0, 2, 3])  # [bs,hd,sl,dim]-> [hd,bs,sl,dim]
    hd_num = input_tensor.get_shape().as_list()[0]
    bs = tf.shape(input_tensor)[1]
    sl = tf.shape(input_tensor)[2]
    hd_dim = input_tensor.get_shape().as_list()[3]

    if merge_var:
      weight = tf.get_variable('W', shape=[hd_num, hd_dim, hn * dup_num])
    else:
      weight_list = []
      for i in range(hd_num):
        sub_weight_list = []
        for j in range(dup_num):
          sub_weight_list.append(tf.get_variable('W_%d_%d' % (i, j), shape=[hd_dim, hn]))
        weight_list.append(tf.concat(sub_weight_list, -1) if dup_num > 1 else sub_weight_list[0])
      weight = tf.stack(weight_list, 0)
    input_tensor_rsp = tf.reshape(input_tensor, [hd_num, bs * sl, hd_dim])  # hd_num, bs*sl, hd_dim
    out_rsp = tf.matmul(input_tensor_rsp, weight)  # hd_num, bs*sl, hn
    if bias:
      if merge_var:
        bias_val = tf.get_variable('bias', shape=[hd_num, 1, hn], dtype=tf.float32,
                                   initializer=tf.constant_initializer(bias_start))
      else:
        bias_list = []
        for i in range(hd_num):
          sub_bias_list = []
          for j in range(dup_num):
            sub_bias_list.append(
                            tf.get_variable(
                                'bias_%d_%d' % (i, j), shape=[1, hn], dtype=tf.float32,
                                initializer=tf.constant_initializer(bias_start)))
          bias_list.append(tf.concat(sub_bias_list, -1) if dup_num > 1 else sub_bias_list[0])
        bias_val = tf.stack(bias_list, 0)
      out_rsp = out_rsp + bias_val
    out = tf.reshape(out_rsp, [hd_num, bs, sl, hn*dup_num])  # [hd,bs,sl,dim]
    return tf.transpose(out, [1, 0, 2, 3])  # [bs,hd,sl,dim]


