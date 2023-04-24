# scNTImpute:single-cell Neural Topic Imputation Model
scNTImpute is a model for imputation of scRNA-seq data, helping to improve downstream analysis of single-cell RNA sequencing data and more. Without any installation, researchers can download and use the source generation to pre-process scRNA-seq data to accurately and efficiently identify detached values and impute them accurately. scNTImpute is fully described and applied in the manuscript "Imputation Methods for Single-Cell RNA-seq Data Using Neural Topic Models". This repository includes instructions for use, operating environment and data format requirements
# Content
[Instructions](#instructions)

[Operating environment](#operating-environment)

[Data format](#data-format)

# Instructions
本模型采用Pyton语言编写，可将单scRNA-seq数据放入train_scNTImpute.py进行训练及插补。若要进行迁移学习可通过另一个文件来进行。启用GPU计算会显著提升性能，请在使用前安装支持GPU的PyTorch。
The model is written in Pyton language and can be trained and imputed by putting single scRNA-seq data into train_scNTImpute.py. To perform transfer learning, this can be done via a separate file. Enabling GPU computing will significantly improve performance, please install PyTorch with GPU support before use.
# Operating environment
python 3.7

PyTorch 1.11

numpy 1.21

anndata 0.8

pandas 1.3

# Data format
scNTImpute需要输入一个细胞-基因表达矩阵，格式为AnnData对象。有关AnnData的详细介绍可在此处查看。
