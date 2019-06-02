# Language Model based Transfer Learning for Natural Language Inference Tasks (Entailment)

* The NLI tasks include the most popular two: [SNLI](https://nlp.stanford.edu/projects/snli/) and [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/)
* All baseline methods and proposed model are based on sentence-encoding, instead of any form of cross-attention at token-level. 
* The programming framework is a little different with the other projects in this repo, mainly in terms of hyperparameter parsing. The `HParams` bulitin Tensorflow is used to manage the hyperparameters. Specifically, each model has a `HParams` to store the corresponding hyperparamters. When the program is running, the `Config.py` will automatically fetch the `HParams` instance and parse the input hyperparameters for the chose model. 

## Extra Python Package Requirement 

* ftfy
* SpaCy >= 2.0.0

## Data Preparation 

There are four parts should be download as follows. For convenience, I packed them in a single zip, download from [google drive](https://drive.google.com/file/d/1X51U2v2Mn4KGXs1wf6f4tDPkApdUe0pD/view?usp=sharing).

1. [40K BPE dictionary](https://github.com/openai/finetune-transformer-lm/tree/master/model)
2. [pretrained language model](https://github.com/openai/finetune-transformer-lm/tree/master/model)
3. [snli dataset](https://nlp.stanford.edu/projects/snli/) 
4. [multinli dataset](https://www.nyu.edu/projects/bowman/multinli/) (with [matched](https://inclass.kaggle.com/c/multinli-matched-open-evaluation) and [mismatched](https://inclass.kaggle.com/c/multinli-mismatched-open-evaluation) test sets from kaggle]. 

### Data Checklist

Note that some original data files may be **re-named** for compatibility with this project.

    ./dataset/bpe/encoder_bpe_40000.json
    ./dataset/bpe/vocab_40000.bpe
    ./dataset/pretrained_transformer/params_[0-9].npy
    ./dataset/pretrained_transformer/params_shapes.json
    ./dataset/snli_1.0/snli_1.0_[train|dev|test].jsonl
    ./dataset/multinli_1.0/multinli_1.0_[train|dev|test]_matched.jsonl
    ./dataset/multinli_1.0/multinli_1.0_[train|dev|test]_mismatched.jsonl


## Model Training:

After cloning this repo
    
    cd Proj_NLI_Trans_mtsa

### For SNLI:

    python3 main.py --network_type mtsa --training_params n_steps=81000 --model_params clf_afn=elu --gpu 0 --model_dir_prefix train_snli

### For MultiNLI Matched

    python main.py --dataset multinli_m --network_type transformer --training_params n_steps=60000 --model_params use_mtsa=True --gpu 0 --model_dir_prefix train_multinli_m

### For MultiNLI Mismatched

    python main.py --dataset multinli_mm --network_type transformer --training_params n_steps=50000 --model_params use_mtsa=True --gpu 0 --model_dir_prefix train_multinli_mm
 
 
## Acknowledge

Many thanks to [this repo](https://github.com/openai/finetune-transformer-lm) for their code and models.