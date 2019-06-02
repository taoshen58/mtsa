# TREC Question-Type Classification with Multi-mask Tensorized Self-Attention

* The detail of the framework of this project is elaborated at the end of `README.md` in the root of this repo.
* Dataset official web site: [http://cogcomp.org/Data/QA/QC/](http://cogcomp.org/Data/QA/QC/)


## Data Preparation

* Please download the 6B pre-trained model to `./dataset/glove/`. The download address is [http://nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)
* This repo has contained the dataset.
**Data files checklist for running with default parameters:**

        ./dataset/glove/glove.6B.300d.txt
        ./dataset/QuestionClassification/train_5500.label.txt
        ./dataset/QuestionClassification/TREC_10.label.txt


## Model Training

after cloning this repo and preparing the data

    cd Proj_TREC_mtsa
    python3 snli_main.py --network_type mtsa --dropout 0.5 --num_steps 99000 --fine_grained False --model_dir_prefix training --gpu 0

The results will appear at the end of training. There several params frequently used:

* `--num_steps`: training step;
* `--eval_period`: change the evaluation frequency/period.
* `--save_model`: [default false] if True, save model to the model ckpt dir;
* `--train_batch_size` and `--test_batch_size`: set to smaller value if GPU memory is limited.
* `--dropout` and `--wd`: dropout keep prob and L2 regularization weight decay.
* `--word_embedding_length` and `--glove_corpus`: word embedding Length and glove model.

## Tips

* the reported performance can be achieved under Tensorflow 1.3.0 with CUDA8. 

