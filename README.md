# BayesID

This repository is designed to contain a number of tools that can be used for Bayesian system identification (ID) such as sampling, filtering, and log marginal likelihood evaluation. The contents of each folder are described below. For more information on how to use Bayesian system ID, please check out our [paper](https://link.springer.com/article/10.1007/s11071-020-05925-8).

## Contents Description

### examples

#### autonomous
This folder contains a collection of numerical experiments on autonomous systems that were used in our 2020 paper [Bayesian system id: optimal management of parameter, model, and measurement uncertainty](https://link.springer.com/article/10.1007/s11071-020-05925-8). These examples have been revised since the time of publishing to be compatible with updated, faster implementations of certain algorithms.

#### non-autonomous
This folder contains a collection of numerical experiments on non-autonomous systems that are used in a recently submitted paper titled [Likelihood-based generalization of Markov parameter estimation and multiple shooting objectives in system identification](https://arxiv.org/abs/2212.13902). Some of the scripts in this folder require saved results or datasets that can be accessed [here](https://drive.google.com/drive/folders/1ReKW6OeJDp201IFtisRdjyD_ZxLlt4pG?usp=drive_link).

### filtering
This folder contains a collection of functions for evaluating the prediction and update steps of various Bayesian filters. These filters are used within the Bayesian system ID algorithm to evaluate the marginal posterior. The included filters are the Kalman filter, the unscented Kalman filter, the ensemble Kalman filter, and the Gauss-Hermite filter.

### logposterior
This folder contains functions for evaluating the log marginal likelihood.

### nlogposterior
This folder contains functions for evaluating the negative log marginal likelihood and its gradients. We distinguish this folder from [logposterior](#logposterior) in order to optimize the log marginal likelihood computation for gradient evaluation.

### plotting
This folder contains a number of functions used for plotting posterior samples and filtering estimates.

### sampling
This folder contains functions for MCMC sampling using the [DRAM](https://link.springer.com/article/10.1007/s11222-006-9438-0) algorithm.

### utils
This folder contains various helper functions that are used throughout the other folders.

## Citation
If you found this repository useful, please consider citing it with the following citation:

Galioto, Nicholas, and Alex Arkady Gorodetsky. "Bayesian system ID: optimal management of parameter, model, and measurement uncertainty." Nonlinear Dynamics 102.1 (2020): 241-267.

<b>BibTeX:</b>
```bibtex
@article{galioto2020bayesian,
  title={Bayesian system {ID}: optimal management of parameter, model, and measurement uncertainty},
  author={Galioto, Nicholas and Gorodetsky, Alex Arkady},
  journal={Nonlinear Dynamics},
  volume={102},
  number={1},
  pages={241--267},
  year={2020},
  publisher={Springer}
}
```

## Information
Author: [Nick Galioto](https://scholar.google.com/citations?user=psGSgNoAAAAJ&hl=en&oi=sra)\
Contact: [ngalioto@umich.edu](mailto:ngalioto@umich.edu)
