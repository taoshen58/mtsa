# Stanford Sentiment Treebank with Multi-Mask

Dataset official web site: [https://nlp.stanford.edu/sentiment/](https://nlp.stanford.edu/sentiment/)

## Data Preparation

* Please download the 6B pre-trained model to `./dataset/glove/`. The download address is [http://nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)
* This repo has contained the dataset.
**Data files checklist for running with default parameters:**

        ./dataset/glove/glove.6B.300d.txt
        ./dataset/stanfordSentimentTreebank/datasetSentences.txt
        ./dataset/stanfordSentimentTreebank/datasetSplit.txt
        ./dataset/stanfordSentimentTreebank/dictionary.txt
        ./dataset/stanfordSentimentTreebank/original_rt_snippets.txt
        ./dataset/stanfordSentimentTreebank/sentiment_labels.txt
        ./dataset/stanfordSentimentTreebank/SOStr.txt
        ./dataset/stanfordSentimentTreebank/STree.txt


## Training model

After repo cloning and data preparation, Just simply run the code:

    cd Proj_SST_mtsa
    python3 sst_main.py --network_type mtsa --fine_grained True --model_dir_prefix training --gpu 0

The results will appear at the end of training. We list several frequent use parameters in training. (Please refer to the README.md in repo root for more details).

* `--num_steps`: training step;
* `--eval_period`: change the evaluation frequency/period.
* `--save_model`: [default false] if True, save model to the model ckpt dir;
* `--train_batch_size` and `--test_batch_size`: set to smaller value if GPU memory is limited.
* `--dropout` and `--wd`: dropout keep prob and L2 regularization weight decay.
* `--word_embedding_length` and `--glove_corpus`: word embedding Length and glove model.
