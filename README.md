Interpretable Graph Convolutional Network for Multi-view Semi-supervised Learning
====
This is the Pytorch implementation of IMvGCN proposed in our paper:

[Zhihao Wu](https://zhihaowu99.github.io/), Xincan Lin, Zhenghong Lin, [Zhaoliang Chen](https://chenzl23.github.io/), Yang Bai and Shiping Wang*, [Interpretable Graph Convolutional Network for Multi-view Semi-supervised Learning](https://ieeexplore.ieee.org/abstract/document/10080867), IEEE Transactions on Multimedia.

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
  * --path: path of datasets.
  * --dataset: name of datasets.
  * --seed: random seed.
  * --fix_seed: fix the seed or not.
  * --n_repeated: number of repeat times.
  * --lr: learning rate.
  * --weight_decay: weight decay.
  * --ratio: label ratio.
  * --num_epoch: number of training epochs.
  * --Lambda: hyperparameter $\lambda$.
  * --alpha: hyperparameter $\alpha$.

All the configs are set as default, so you only need to set dataset.
For example:

 ```
 python main.py --dataset 3Sources
 ```

## Dataset

Please unzip the datasets folders first.

Saved in ./data/datasets/datasets.7z

*Run construct_lp.py to generate laplacian matrices. Data splitting function can be found in utils.py.*

*Please feel free to email me for the four large datasets or any questions.*

## Reference

```
@article{10080867,
  author={Wu, Zhihao and Lin, Xincan and Lin, Zhenghong and Chen, Zhaoliang and Bai, Yang and Wang, Shiping},
  journal={IEEE Transactions on Multimedia}, 
  title={Interpretable Graph Convolutional Network for Multi-View Semi-Supervised Learning}, 
  year={2023},
  volume={25},
  number={},
  pages={8593-8606},
  doi={10.1109/TMM.2023.3260649}}
```
