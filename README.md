# MSFlow: Multi-Scale Normalizing Flows for Unsupervised Anomaly Detection

This is an official implementation of "MSFlow: Multi-Scale Normalizing Flows for Unsupervised Anomaly Detection".

## Abstract

Unsupervised anomaly detection (UAD) attracts a lot of research interest and drives widespread applications, where only anomaly-free samples are available for training. Some UAD applications even intend to locate the anomalous regions further without any anomaly information. Although the absence of anomalous samples and annotations deteriorates the UAD performance, an inconspicuous yet powerful statistics model, the normalizing flows, is appropriate for anomaly detection and localization in the unsupervised fashion. Only trained on the anomaly-free data, the flow-based probabilistic models can efficiently distinguish unpredictable anomalies by assigning them much lower likelihoods than normal data.Nevertheless, the size variation of the unpredictable anomalies introduces another inconvenience to the flow-based methods for high-precision anomaly detection and localization. To generalize the anomaly size variation, a novel **M**ulti-**S**cale **Flow**s-based framework dubbed **MSFlow** composed of asymmetrical parallel flows followed by a fusion flow to exchange multi-scale perceptions is proposed. Moreover, we adopt different multi-scale aggregation strategies for the image-wise anomaly detection and pixel-wise anomaly localization according to the discrepancy between them. On the challenging MVTec AD benchmark, our MSFlow achieves a new state-of-the-art with detection AUORC score of 99.7%, localization AUROC score of 98.8% and PRO score of 97.1%.

![The framework of MSFlow](./imgs/framework.png)

## Results on the MVTec AD benchmark

| Classes             | Det. AUROC | Loc. AUROC | Loc. PRO |
| ------------------- | :--------: | :--------: | :------: |
| Carpet              |   100.0    |    99.4    |   99.6   |
| Grid                |    99.8    |    99.4    |   99.1   |
| Leather             |   100.0    |    99.7    |   99.9   |
| Tile                |   100.0    |    98.2    |   95.3   |
| Wood                |   100.0    |    97.1    |   96.6   |
| Bottle              |   100.0    |    99.0    |   98.5   |
| Cable               |    99.5    |    98.5    |   93.7   |
| Capsule             |    99.2    |    99.1    |   98.4   |
| Hazelnut            |   100.0    |    98.7    |   96.6   |
| Metal Nut           |   100.0    |    99.3    |   97.6   |
| Pill                |    99.6    |    98.8    |   96.0   |
| Screw               |    97.8    |    99.1    |   94.2   |
| Toothbrush          |   100.0    |    98.5    |   91.6   |
| Transistor          |   100.0    |    98.3    |   99.8   |
| Zipper              |   100.0    |    99.2    |   99.4   |
| **Overall Average** |  **99.7**  |  **98.8**  | **97.1** |
