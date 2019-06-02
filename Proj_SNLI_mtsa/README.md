# Natural Language Inference with Multi-mask Tensorized Self-Attention

* The detail of the framework of this project is elaborated at the end of `README.md` in the root of this repo.


## Data Preparation
* Please download the 6B pre-trained model to ./dataset/glove/. The download address is http://nlp.stanford.edu/data/glove.6B.zip
* Please download the SNLI dataset files from [https://nlp.stanford.edu/projects/snli/snli_1.0.zip](https://nlp.stanford.edu/projects/snli/snli_1.0.zip).


**Data files checklist for running with default parameters:**

    ./dataset/glove/glove.6B.300d.txt
    ./dataset/snli_1.0/snli_1.0_train.jsonl
    ./dataset/snli_1.0/snli_1.0_dev.jsonl
    ./dataset/snli_1.0/snli_1.0_test.jsonl


## Model Training

after cloning this repo

    cd Proj_SNLI_mtsa
    python3 snli_main.py --network_type mtsa --model_dir_prefix training --gpu 0

The results will appear at the end of training. There several params frequently used:

* `--num_steps`: training step;
* `--eval_period`: change the evaluation frequency/period.
* `--save_model`: [default false] if True, save model to the model ckpt dir;
* `--train_batch_size` and `--test_batch_size`: set to smaller value if GPU memory is limited.
* `--dropout` and `--wd`: dropout keep prob and L2 regularization weight decay.
* `--word_embedding_length` and `--glove_corpus`: word embedding Length and glove model.
* `--load_model`: if a pre-trained model is needed for further fine-tuning, set this parameter to True
* `--load_path`: specify a ckpt path to load


## Model Test

To test a model with a ckpt file, simply run

    python3 snli_main.py --mode test --network_type mtsa --load_model True --load_path path-to-ckpt/model.ckpt --test_batch_size 100 --model_dir_suffix test_mode --gpu 0


