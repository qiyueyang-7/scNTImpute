# scNTImpute:single-cell Neural Topic Imputation Model
scNTImpute is a model for imputation of scRNA-seq data, helping to improve downstream analysis of single-cell RNA sequencing data and more. Without any installation, researchers can download and use the source code to pre-process scRNA-seq data to accurately and efficiently identify detached values and impute them accurately. scNTImpute is fully described and applied in the manuscript "Imputation Methods for Single-Cell RNA-seq Data Using Neural Topic Models". This repository includes instructions for use, operating environment and data format requirements.
# Content
[Instructions](#instructions)

[Operating environment](#operating-environment)

[Data format](#data-format)

# Introduction
The model is written in Pyton language and can be trained and imputed by putting single scRNA-seq data into `train_scNTImpute.py`(script file in folder scripts). To perform transfer learning, this can be done via a separate file. Enabling GPU computing will significantly improve performance, please install PyTorch with GPU support before use.(For other introductions, please refer to `readme.txt`.)

# Operating environment
Python 3.7

PyTorch 1.11

Numpy 1.21

Anndata 0.8

Pandas 1.3

# Data format
scNTImpute requires the input of a cell-gene expression matrix in the format of an AnnData object, whereas the scRNA-seq dataset format is usually `.csv`or other forms. For the smoothness and convenience of use, we have completed the data conversion step in the main code (script file) . Simply pass in scRNA-seq data (e.g. csv files) normally. A detailed description of AnnData can be found [here](https://anndata.readthedocs.io/en/latest/). All datasets used are public datasets and the sources of the datasets can be viewed in detail in the manuscript.
## Output file
If the user manipulates our script file, a results folder is created when it is run, which includes numerical results for each evaluation metric in the manuscript (A log file: record other parameters and information at the same time) , pdf files of the different epoch clustering results, saving of the model parameters (if the trained model needs to be saved) and some others.

# License
The use of scNTInput follows the MIT License
