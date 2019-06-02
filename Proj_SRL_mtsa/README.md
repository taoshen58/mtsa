# Semantic Role Labeling with Multi-mask Tensorized Self-Attention (MTSA)

## Requirement

1. Python2
2. Tensorflow>=1.3

## Getting Started

1. Applying MTSA to SRL tasks is based on the SRL Tagger released by @XMUNLP at [here](https://github.com/XMUNLP/Tagger). Please clone the codes first and follow the `README.md` in that repo to know how the code works.
2. Copy the [`deepatt.py`](deepatt.py) in this repo to folder [`Tagger/models`](https://github.com/XMUNLP/Tagger/tree/master/models) and then replace original one.
3. Add a new variable initialization method, Glorot, after [line 306](https://github.com/XMUNLP/Tagger/blob/47cb49c8ac79b4b6932a9ed3b2c80699546c585c/main.py#L306) in [`Tagger/main.py`](https://github.com/XMUNLP/Tagger/blob/master/main.py) by using following lines. The reason why the glorot is used is that we empirically found that the orthogonal initializer led to the NaN loss in tensorflow>=1.5

        elif params.initializer == "glorot":
            return tf.glorot_normal_initializer()
        
4. Do not forget to choose the 'glorot' as the initializer by using running command 

        --training_params=...,initializer=orthogona,...
    
to replace original `initializer=orthogonal` in [Training Command](https://github.com/XMUNLP/Tagger#training-and-validating).

## Tips to Run the [SRL-Tagger](https://github.com/XMUNLP/Tagger)

1. The validation script (i.e., `run.sh`) is not provided in the Repo but is indispensable when running the code. You can save the codes given in [this Section](https://github.com/XMUNLP/Tagger#training-and-validating) to a new `run.sh` file. Besides, do not forget to check the arguments passed to `run.sh` in [Tagger/utils/validation.py](https://github.com/XMUNLP/Tagger/blob/master/utils/validation.py).
2. Read After Run Codes Successfully There are a bug in Tagger and I give you a solution here. The bug is that the validation program need to read the latest checkpoint multiple times because because `tf.contrib.learn.Estimator.predict` need to re-load the checkpoint when it is invoked for every validation batch. However, when validation bash script is running in a sub-process, the latest checkpoint model could be updated by the main training process. This may lead to the situation that the latest checkpoint updates during a validation procedure, resulting in wrong validation results. The solution is that, before starting the period validation, you can copy the latest ckpt to a temporary path (do not forget to form a new tensorflow `checkpoint` file in the path) and then pass the path to the `run.sh`.

## Contact Info

Please feel free to open an issue if you encounter any bug and confusion when you execute the codes.

## Acknowledgements

Thanks to [Zhixing Tan](mailto:playinf@stu.xmu.edu.cn) for the [SRL Tagger Framework](https://github.com/XMUNLP/Tagger) and his neat code style!


