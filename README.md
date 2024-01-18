# NOCCO_Shapley_values

# A preprocessing Shapley value-based approach to detect relevant and disparity prone features in machine learning

## Introduction

This work proposes an approach to detect relevant and disparity prone features, before training step, in machine learning. Our proposal is based on the normalized version of Hilbert-Schmidt Independence Criterion (HSIC), called NOCCO, and the hypothesis is that features with high marginal dependence with the outcomes may have a high impact into the both algorithm predictive power and disparate results. In order to evaluate our proposal, we consider the following datasets:

- Adult Income: https://archive.ics.uci.edu/ml/datasets/adult
- COMPAS Recidivism risk: https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing
- LSAC: http://www.seaphe.org/databases.php
- Rice: https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik
- Banknotes: https://archive.ics.uci.edu/dataset/267/banknote+authentication
- Red Wine Quality: https://archive.ics.uci.edu/dataset/186/wine+quality
- Diabetes Pima: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- Raisin: https://archive.ics.uci.edu/dataset/850/raisin

All codes and data files can be accessed from this repository.
