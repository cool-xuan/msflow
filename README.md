# MSFlow: Multi-Scale Normalizing Flows for Unsupervised Anomaly Detection

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/msflow-multi-scale-flow-based-framework-for/anomaly-detection-on-mvtec-ad)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad?p=msflow-multi-scale-flow-based-framework-for)

This is an official implementation of "[MSFlow: Multi-Scale Normalizing Flows for Unsupervised Anomaly Detection](https://arxiv.org/pdf/2308.15300v1.pdf)".

## Inmportant Notice

- [2024-01-11] We have extended our code implementation to the [VisA dataset](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar). AMP of pyTorch is supported in the updated version, which can accelerate the training process. Besides, the log files on the MVTec AD dataset and VisA dataset are also provided for reference (```log_mvtec.txt``` and ```log_visa.txt```).

- [2023-12-11] 🎉 Our paper has been accepted by *TNNLS 2024*, and the formal citation will be updated soon.

- [2023-09-23] We have updated the [paper](https://arxiv.org/pdf/2308.15300v1.pdf) and code to the full version, which supports the MVTec AD dataset and achieves SOTA performance. 

## Abstract

Unsupervised anomaly detection (UAD) attracts a lot of research interest and drives widespread applications, where only anomaly-free samples are available for training. Some UAD applications intend to locate the anomalous regions further even without any anomaly information. Although the absence of anomalous samples and annotations deteriorates the UAD performance, an inconspicuous yet powerful statistics model, the normalizing flows, is appropriate for anomaly detection and localization in an unsupervised fashion. The flow-based probabilistic models, only trained on anomaly-free data, can efficiently distinguish unpredictable anomalies by assigning them much lower likelihoods than normal data. Nevertheless, the size variation of unpredictable anomalies introduces another inconvenience to the flow-based methods for high-precision anomaly detection and localization. To generalize the anomaly size variation, we propose a novel **M**ulti-**S**cale **Flow**s-based framework dubbed **MSFlow** composed of asymmetrical parallel flows followed by a fusion flow to exchange multi-scale perceptions. Moreover, different multi-scale aggregation strategies are adopted for the image-wise anomaly detection and pixel-wise anomaly localization according to the discrepancy between them. On the challenging MVTec AD benchmark, our MSFlow achieves a new state-of-the-art with detection AUORC score of 99.7%, localization AUROC score of 98.8% and PRO score of 97.1%.

![The framework of MSFlow](./imgs/framework.png)

## Enviroment

- Python 3.9
- scikit-learn
- scikit-image
- PyTorch >= 1.10
- CUDA 11.3
- [FrEIA](https://github.com/VLL-HD/FrEIA) (Please install FrEIA following the [official installation](https://github.com/VLL-HD/FrEIA#table-of-contents))

## Prepare datasets

It is recommended to symlink the dataset root to `$msflow/data`.
If your folder structure is different, you may need to change the corresponding paths in `default.py`.

**For MVTec AD data**, please download from [MVTec AD download](https://www.mvtec.com/company/research/datasets/mvtec-ad). Download and extract them to `$msflow/data`, and make them look like the following data tree:

```shell
MVTec
├── bottle
│   ├── ground_truth
│   │   ├── broken_large
│   │   └── ...
│   ├── test
│   │   ├── good
│   │   ├── broken_large
│   │   └── ...
│   └── train
│       └── good
├── cable
└── ...
```

**For VisA data**, please download from [VisA download](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar). Download and extract them to `$msflow/data`, and make them look like the following data tree:

```shell
VisA
├── candle
│   ├── ground_truth
│   │   └── bad
│   ├── test
│   │   ├── bad
│   │   └── good
│   └── train
│       └── good
├── capsules
└── ...
```

Thanks [spot-diff](https://github.com/amazon-science/spot-diff/tree/main) for providing the code to reorganize the VisA dataset in MVTec AD format. For more details, please refer to this [data preparation guide](https://github.com/amazon-science/spot-diff/tree/main#data-preparation).

## Training and Testing

All checkpoints will be saved to the working directory, which is specified by `work_dir` in the `default` file.

By default, we evaluate the model on the test set after each meta epoch, you can change the pro evaluation interval by modifying the interval argument in the shell or `default` file.

### Training

For MVTec AD dataset:
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --mode train \
    --dataset mvtec --class-names all
```

For VisA dataset:
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --mode train \
    --dataset visa --class-names all --pro-eval
```

### Testing

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --mode test --class-name bottle --eval_ckpt $PATH_OF_CKPT 
```


## Results on the MVTec AD benchmark

| Classes             | Det. AUROC | Loc. AUROC |
| ------------------- | :--------: | :--------: |
| Carpet              |   100.0    |    99.4    |
| Grid                |    99.8    |    99.4    |
| Leather             |   100.0    |    99.7    |
| Tile                |   100.0    |    98.2    |
| Wood                |   100.0    |    97.1    |
| Bottle              |   100.0    |    99.0    |
| Cable               |    99.5    |    98.5    |
| Capsule             |    99.2    |    99.1    |
| Hazelnut            |   100.0    |    98.7    |
| Metal Nut           |   100.0    |    99.3    |
| Pill                |    99.6    |    98.8    |
| Screw               |    97.8    |    99.1    |
| Toothbrush          |   100.0    |    98.5    |
| Transistor          |   100.0    |    98.3    |
| Zipper              |   100.0    |    99.2    |
| **Overall Average** |  **99.7**  |  **98.8**  |

## Results on the VisA benchmark

| Classes             | Det. AUROC | Loc. AUROC |
| ------------------- | :--------: | :--------: |
| candle              |    97.7    |    98.3    |
| capsules            |    98.0    |    96.2    |
| cashew              |    94.9    |    98.7    |
| chewinggum          |    93.6    |    99.7    |
| fryum               |    88.2    |    99.6    |
| macaroni1           |    97.6    |    97.6    |
| macaroni2           |    98.0    |    89.5    |
| pcb1                |    96.0    |    98.9    |
| pcb2                |    93.5    |    97.8    |
| pcb3                |    94.4    |    98.9    |
| pcb4                |    93.0    |    99.5    |
| pipe_fryum          |    97.0    |    98.9    |
| **Overall Average** |  **95.2**  |  **97.8**  |

## Thanks to

- [FrEIA](https://github.com/VLL-HD/FrEIA)
- [CFlow-AD](https://github.com/gudovskiy/cflow-ad)
- [CSFlow](https://github.com/marco-rudolph/cs-flow)
- [spot-diff](https://github.com/amazon-science/spot-diff/tree/main)

## Citation

If you find this work useful for your research, please cite our paper. The formal citation of TNNLS will be updated soon.

```bibtex
@article{zhou2023msflow,
  title={MSFlow: Multi-Scale Flow-based Framework for Unsupervised Anomaly Detection},
  author={Zhou, Yixuan and Xu, Xing and Song, Jingkuan and Shen, Fumin and Shen, Heng Tao},
  journal={arXiv preprint arXiv:2308.15300},
  year={2023}
}
```
