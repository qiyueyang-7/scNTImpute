# scNTImpute:single-cell Neural Topic Imputation Model
scNTImpute is a model for imputation of scRNA-seq data, helping to improve downstream analysis of single-cell RNA sequencing data and more. Without any installation, researchers can download and use the source generation to pre-process scRNA-seq data to accurately and efficiently identify detached values and impute them accurately. scNTImpute is fully described and applied in the manuscript "Imputation Methods for Single-Cell RNA-seq Data Using Neural Topic Models". This repository includes instructions for use, operating environment and data format requirements
# Content
[Instructions](#instructions)

[Operating environment](#operating-environment)

[3 XY](#xy)

# Instructions
本模型采用Pyton语言编写，可将单scRNA-seq数据放入train_scNTImpute.py进行训练及插补。若要进行迁移学习可通过另一个文件来进行。启用GPU计算会显著提升性能，请在使用前安装支持GPU的PyTorch。手稿"Imputation Methods for Single-Cell RNA-seq Data Using Neural Topic Models"的源代码。

# Operating environment
python 3.7

PyTorch 1.11

numpy 1.21

anndata 0.8

pandas 1.3

# 3 XY
