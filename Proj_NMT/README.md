# Neural Machine Translation with Multi-mask Tensorized Self-Attention (MTSA)

## Requirement

python3
tensorflow=1.9.0
tensor2tensor==1.9.0

## Getting Started

Following procedures are introduction to integrating MTSA to official Transformer implementation.

1. clone `https://github.com/tensorflow/tensor2tensor/tree/v1.9.0` to your local pc or server.
2. copy `./files/mtsa.py` to `./tensor2tensor/tensor2tensor/layers/`
3. Replace `./tensor2tensor/tensor2tensor/layers/common_attention.py` with `./files/common_attention.py`. Specifically, the only difference is that a code piece for invoking MTSA is added in around line 3474.
4. Replace `./tensor2tensor/tensor2tensor/models/transformer.py` with `./files/transformer.py`. Specifically, the differences are:
    * in `transformer_encoder`, add extra arguments passed into `common_attention.multihead_attention` around line 1224-1233, 
    * in `transformer_decoder`, add extra arguments passed into `common_attention.multihead_attention` around line 1310-1363 (these are for the attention from the decoder to encoder and decoder self-attention)
    * Add parameters parsing items for MTSA around line 1541
5. The deployment is done, the hyper-parameters are:
    ```
    hparams.add_hparam("encoder_self_attention_type", "none")
    hparams.add_hparam("decoder_self_attention_type", "none")
    hparams.add_hparam("decoder_encoder_attention_type", "none")
    hparams.add_hparam("use_k_mtsa", False)
    hparams.add_hparam("afn_extra", "none")
    hparams.add_hparam("afn_dot", "exp")
    hparams.add_hparam("afn_multi", "scaled_sigmoid")
    hparams.add_hparam("bias_start", 0.)
    hparams.add_hparam("bi_direction", False)
    ```
6. Simply add the parameters to `--hparams` for running with an specific set of hyper-parameters. for example, `--hparams=eencoder_self_attention_type=mtsa,afn_multi=scaled_sigmoid,batch_size=2048,`
7. Set `--worker_gpu` for specific number of gpus to run

## Train the model

1. Follow the official instruction for Transformer in `tensor2tensor`, https://github.com/tensorflow/tensor2tensor/blob/v1.9.0/README.md, to run prepare the dataset and generate proper TFDataset.
2. Use the specific parameter corresponding to MTSA self-attention mechanism.

### Training Setup 1 
```
t2t-trainer \
  --local_eval_frequency 500000 \
  --eval_throttle_seconds 200000 \
  --save_checkpoints_secs 2000 \
  --keep_checkpoint_max 5 \
  --data_dir=${PRE_PROCESSED_DATA_DIR} \
  --tmp_dir=~/t2t/nmt/tmp \
  --problem=translate_ende_wmt32k \
  --model=transformer \
  --hparams_set=transformer_base_single_gpu \
  --output_dir=${OUTPUT_DIR} \
  --hparams=encoder_self_attention_type=mtsa,afn_multi=scaled_sigmoid,bias_start=5.,\
  --worker_gpu=1
```

### Training Setup 2
```
t2t-trainer \
  --local_eval_frequency 500000 \
  --eval_throttle_seconds 200000 \
  --save_checkpoints_secs 550 \
  --keep_checkpoint_max 16 \
  --data_dir=${PRE_PROCESSED_DATA_DIR} \
  --tmp_dir=${TMP_DATA_DIR} \
  --problem=translate_ende_wmt32k \
  --model=transformer \
  --hparams_set=transformer_base \
  --output_dir=${OUTPUT_DIR} \
  --hparams=encoder_self_attention_type=mtsa,batch_size=6144 \
  --worker_gpu=4 \
  --train_steps 133000 
```
    
    






