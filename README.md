# BayesID

This repository is designed to contain a number of tools that can be used for Bayesian system identification (ID) such as sampling, filtering, and log marginal likelihood evaluation. The contents of each folder are described below. For more information on how to use Bayesian system ID, please check out our [paper](https://link.springer.com/article/10.1007/s11071-020-05925-8).

## Contents Description

### examples

#### autonomous
This folder contains a collection of numerical experiments on autonomous systems that were used in our 2020 paper [Bayesian system id: optimal management of parameter, model, and measurement uncertainty](https://link.springer.com/article/10.1007/s11071-020-05925-8). These examples have been revised since the time of publishing to be compatible with updated, faster implementations of certain algorithms.

#### non-autonomous
Coming soon. This folder will contain a collection of numerical experiments on non-autonomous systems that are used in an upcoming paper.

### filtering
This folder contains a collection of functions for evaluating the prediction and update steps of various Bayesian filters. These filters are used within the Bayesian system ID algorithm to evaluate the marginal posterior. The included filters are the Kalman filter and the unscented Kalman filter.

### logposterior
This folder contains functions for evaluating the log marginal likelihood.

### nlogposterior
This folder contains functions for evaluating the negative log marginal likelihood and its gradients. We distinguish this folder from [logposterior](#logposterior) in order to optimize the log marginal likelihood computation for gradient evaluation.

### plotting
This folder contains a number of functions used for plotting posterior samples and filtering estimates.

### sampling
This folder contains functions for MCMC sampling using the DRAM algorithm.

### utils
This folder contains various helper functions that are used throughout the other folders.

## Information
Author: [Nick Galioto](https://scholar.google.com/citations?user=psGSgNoAAAAJ&hl=en&oi=sra)\
Contact: [ngalioto@umich.edu](mailto:ngalioto@umich.edu)