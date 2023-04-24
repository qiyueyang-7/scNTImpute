# scNTImpute:single-cell Neural Topic Imputation Model
scNTImpute是针对scRNA-seq数据进行插补的模型，有助提高单细胞RNA测序数据的下游分析等。无需任何安装，研究人员可以通过下载使用源代，对但scRNA-seq数据进行预处理，可以精准并有效地识别出脱落值，并对其进行准确插补。scNTImpute的完整描述及应用可在手稿"Imputation Methods for Single-Cell RNA-seq Data Using Neural Topic Models"中获得。这个repository包括使用说明，运行环境及数据格式等要求
# 内容
[1 x](#1 x)

[2 运行环境](#2 运行环境)

[3 数据格式](#readme)

# 1 x
本模型采用Pyton语言编写，可将单scRNA-seq数据放入train_scNTImpute.py进行训练及插补。若要进行迁移学习可通过另一个文件来进行。启用GPU计算会显著提升性能，请在使用前安装支持GPU的PyTorch。手稿"Imputation Methods for Single-Cell RNA-seq Data Using Neural Topic Models"的源代码。

# 2 运行环境
python 3.7

PyTorch 1.11

numpy 1.21

anndata 0.8

pandas 1.3

# 3 数据格式
