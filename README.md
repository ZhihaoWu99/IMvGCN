Interpretable Graph Convolutional Network for Multi-view Semi-supervised Learning
====
This is the Pytorch implementation of IMvGCN proposed in our paper:

Zhihao Wu, Xincan Lin, Zhenghong Lin, Zhaoliang Chen, Yang Bai and Shiping Wang*, [Interpretable Graph Convolutional Network for Multi-view Semi-supervised Learning](https://github.com/ZhihaoWu99/IMvGCN), IEEE Transactions on Multimedia.

![framework](./framework.png)

## Requirement

  * Python == 3.9.12
  * PyTorch == 1.11.0
  * Numpy == 1.21.5
  * Scikit-learn == 1.1.0
  * Scipy == 1.8.0
  * Texttable == 1.6.4
  * Tqdm == 4.64.0

## Usage

```
python main.py
```

  * --device: gpu number or 'cpu'.
  * --path: datasets.
  * --dataset: name of datasets.
  * --seed: random seed.
  * --fix_seed: fix the seed or not.
  * --n_repeated: number of repeated times.
  * --lr: learning rate.
  * --weight_decay: weight decay.
  * --ratio: label ratio.
  * --num_epoch: number of training epochs.
  * --Lambda: hyperparameter.
  * --alpha: hyperparameter.

All the configs are set as default, so you only need to set dataset.
For example:

 ```
 python main.py --dataset 3Sources
 ```

## Datasets

Please unzip the datasets folders first.

Saved in .data/datasets/datasets.7z

