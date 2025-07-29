# MeanFlows_Compare

## Mathematical Principles

Mathematical Principles
This project implements and compares two approaches: MeanFlow and Additive MeanFlow, both of which are based on the concept of mean flow in dynamical systems. The following sections detail the mathematical principles underlying these methods.

### MeanFlow

MeanFlow represents the average velocity of a system over a time interval $[r, t]$, where $r, t \in [0, 1]$ and $r \leq t$. Its mathematical definition is:
$$u(z_t, r, t) = \frac{1}{t - r} \int_r^t v(z_\tau, \tau) , d\tau$$
Here, $z_t$ denotes the state of the system at time $t$, and $v(z_\tau, \tau)$ is the instantaneous velocity at state $z_\tau$ and time $\tau$. This definition captures the average behavior of the system over the specified interval, serving as a cornerstone for modeling time-dependent dynamics.

### Mean Flow Identity

The Mean Flow Identity is a first-order partial differential equation that relates the mean flow $u(z_t, r, t)$ to the instantaneous velocity $v(z_t, t)$ and its rate of change:
$$u(z_t, r, t) = v(z_t, t) - (t - r) \frac{\partial u(z_t, r, t)}{\partial t}$$
This identity can be derived by differentiating the mean flow definition. Let $U(z_t, r, t) = (t - r) u(z_t, r, t)$, then:
$$U(z_t, r, t) = \int_r^t v(z_\tau, \tau) , d\tau$$
Differentiating with respect to $t$ using the Leibniz rule gives:
$$\frac{\partial}{\partial t} U(z_t, r, t) = v(z_t, t)$$
Expanding the left-hand side:
$$\frac{\partial}{\partial t} \left[ (t - r) u(z_t, r, t) \right] = (t - r) \frac{\partial u(z_t, r, t)}{\partial t} + u(z_t, r, t)$$
Thus:
$$(t - r) \frac{\partial u(z_t, r, t)}{\partial t} + u(z_t, r, t) = v(z_t, t)$$
Rearranging yields the Mean Flow Identity. This equation is enforced during training via a loss function:
$$\text{Loss}_{\text{MFI}} = \mu \left| u(z_t, r, t) - \left[ v(z_t, t) - (t - r) \frac{\partial u(z_t, r, t)}{\partial t} \right] \right|^2$$
where $\mu$ is a weighting factor.

### Additive MeanFlow

Additive MeanFlow extends the MeanFlow framework by introducing an additivity constraint, ensuring that the mean flow over an interval can be consistently decomposed into subintervals. The general additivity property states that for any $r < s < t$:
$$(s - r) u(z_r, r, s) + (t - s) u(z_s, s, t) = (t - r) u(z_r, r, t)$$
In this project, we employ a midpoint splitting approach, defining $\text{mid} = \frac{r + t}{2}$, and introduce a state update:
$$z_{\text{mid}} = z_t - (t - \text{mid}) u(z_t, t, \text{mid})$$
The additivity constraint is simplified to:
$$2 u(z_t, t, r) = u(z_t, t, \text{mid}) + u(z_{\text{mid}}, \text{mid}, r)$$
This constraint is enforced through a loss function:
$$\text{Loss}_{\text{additivity}} = \left|  u(z_t, t, r) - \frac{1}{2}\left[ u(z_t, t, \text{mid}) + u(z_{\text{mid}}, \text{mid}, r) \right] \right|^2$$
 This "shortcut" approach uses the midpoint to simplify computations while preserving the core additivity property.

### State Evolution

The state update formula $z_{\text{mid}} = z_t - (t - \text{mid}) u(z_t, t, \text{mid})$ resembles a backward Euler step, reflecting the state change from time $t$ to $\text{mid}$. This update ensures the applicability of the additivity constraint in state-dependent systems.

### Training Implementation

This project compares two models:
**MeanFlow Model**: Trained using only the Mean Flow Identity loss ($\text{Loss}_{\text{MFI}}$).
**Additive MeanFlow Model**: Trained using both the Mean Flow Identity loss and the additivity loss ($\text{Loss}_{\text{MFI}} + \text{Loss}_{\text{additivity}}$).

The additivity constraint is expected to improve prediction accuracy for $t \neq r$ by ensuring the mean flow respects the system's structure, particularly when the dataset includes diverse $t - r$ values.

By incorporating the additivity constraint, this project provides a clear and mathematically rigorous framework for studying the performance differences between MeanFlow and Additive MeanFlow.

## Python Version

This project requires **Python 3.11**.

## Main Dependencies

The main dependencies are listed in `requirements.txt`:

- accelerate==1.9.0
- einops==0.8.1
- matplotlib==3.10.3
- numpy==2.3.2
- scipy==1.16.1
- timm==1.0.19
- torch==2.7.1
- torchvision==0.22.1

## Dataset Description

This project supports the following datasets:

- **MNIST**: Handwritten digit recognition dataset.
- **CIFAR-10**: 60,000 32x32 color images in 10 classes.
- **Fashion MNIST**: Zalando's article images, similar format to MNIST.

Instructions for downloading and preprocessing these datasets will be provided in later sections.

## Main Modules

### 1. `models/DiT.py` — DiT Neural Network

The DiT (Diffusion Transformer) network is a transformer-based architecture designed for image generation tasks. It leverages patch embedding, timestep and label embedding, and a series of transformer blocks with adaptive layer normalization. The DiT model supports conditional and unconditional generation, making it flexible for various datasets and tasks. Its modular design allows for easy integration of new conditioning signals and output layers.

### 2. `models/meanflows.py` — MeanFlow & AdditiveMeanFlow

The MeanFlow module implements a novel generative flow model for images. It supports both conditional and unconditional training, and introduces a flexible normalization strategy. The newly developed AdditiveMeanFlow extends the original MeanFlow by incorporating a consistency loss, which encourages the model outputs to be more stable and robust across different time steps. This consistency loss is combined with the original JVP-based loss to improve training dynamics and sample quality.

### 3. `utils/` — Utilities

- **data_util.py**: Provides dataset loading and preprocessing functions for MNIST, CIFAR-10, and FashionMNIST.
- **fid_util.py**: Implements FID (Fréchet Inception Distance) evaluation for measuring the quality of generated images.
- **vis_util.py**: Contains visualization tools for training curves and experiment reports.
- **ema_util.py**: Maintains an Exponential Moving Average (EMA) of model parameters to stabilize training and improve evaluation results.

## Training Process

The training process is fully automated. Simply run the following command in the project root:

```bash
cd MeanFlows_Compare
bin bash run.sh
```

All steps including data preparation, model training, evaluation, and result saving are handled automatically by the script.

## References

Relevant papers:

1. Mean Flows for One-step Generative Modeling, Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, Kaiming He, [https://arxiv.org/abs/2505.13447]
2. SplitMeanFlow: Interval Splitting Consistency in Few-Step Generative Modeling, Yi Guo, Wei Wang, Zhihang Yuan, Rong Cao, Kuan Chen, Zhengyang Chen, Yuanyuan Huo, Yang Zhang, Yuping Wang, Shouda Liu, Yuxuan Wang, [https://arxiv.org/abs/2507.16884]
3. One Step Diffusion via Shortcut Models, Kevin Frans, Danijar Hafner, Sergey Levine, Pieter Abbeel, [https://arxiv.org/abs/2410.12557]

## License

This project is licensed under the MIT License. See the LICENSE file for details.
