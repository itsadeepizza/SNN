# Spiking Neural Network

Our objective was to tackle some of the classical machine learning (ML) problems through the use of spiking neural networks:

- The **iris dataset**, to gain familiarity with the technique.
- The **MNIST dataset**, to validate our approach.

We also made a decision to implement our solution without the use of specific libraries, instead exclusively utilizing `numpy` and `cython`.

### R-STDP

Our implementation of R-STDP was mainly based on [SVRS] (particularly for Gaussian encoding) and [BBJHCK] (for reinforcement aspects). We did, however, introduce some changes, such as opting not to use the  eligibility trace.

Updating the weights presented some difficulties due to the computationally intensive nature of the operation, prompting us to carry out this process in cython (`fastspike/fastspike.py`).

Our deployment of R-STDP on the Iris dataset (`spiking_iris_R_STDP.py`) proved effective, as we achieved an accuracy of over 90%, which is consistent with the results from more traditional methods given the dataset's limited size.

Attempts to generalize R-STDP with various hidden layers did not yield successful outcomes (`spiking_iris_R_STDP_multilayer.py`).

### Spiking Encoding

The remainder of our code is aimed at digit recognition on the MNIST dataset (https://deepai.org/dataset/mnist). We began by focusing on encoding, experimenting with several methods:

- An encoding technique borrowed from [VCS] (`mnist_encoder.py`);
- A more straightforward encoding method where the likelihood of each pixel firing a spike is proportional to its intensity (`mnist_encoder_simple.py`);
- A hybrid approach, which leverages the technique mentioned in the paper but omits the initial feature extraction stage (`mnist_encoder_bins.py`).

### BP-STDP

To modify R-STDP for use with multiple hidden layers, we implemented BP-STDP following the methodology outlined by [TM].

Our application of BP-STDP on the Iris dataset (`BP_STDP_iris.py`) seems to perform better than R-STDP, with improved convergence in a shorter timeframe.

However, the results on the MNIST dataset (`BP_STDP_mnist.py`) are far from the state of the art.

Despite the use of different encoding strategies and hyperparameter tuning attempts with `optuna` (`BP_STDP_mnist_optuna.py`), the highest accuracy achieved remained at 50%.

## Bibliography

[SVRS] A. Sboev, D. Vlasov, R. Rybka, A. Serenko: *Solving a classification task by spiking neurons with STDP and temporal coding* Procedia Computer Science, Vol 123,
pp 494-500, 2018
https://www.sciencedirect.com/science/article/pii/S1877050918300760)

[BBJHCK] Z. Bing, I. Baumann, Z. Jiang, K. Huang, C. Cai, A. Knol: *Supervised Learning in SNN via Reward-Modulated Spike-Timing-Dependent Plasticity for a Target Reaching Vehicle*, Frontiers in Neurorobotics, Vol. 13, 2019
https://www.frontiersin.org/articles/10.3389/fnbot.2019.00018  


[VCS] R. Vaila, J. Chiasson, V. Saxena: *Deep Convolutional Spiking Neural Networks for Image Classification*, 2019
https://arxiv.org/abs/1903.12272

[TM] A. Tavanaei, A. S. Maida: *BP-STDP: Approximating Backpropagation using Spike Timing Dependent Plasticity*, 2017
https://arxiv.org/abs/1711.04214