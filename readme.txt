#scNTImpute:single-cell Neural Topic Imputation Model
scNTImpute is a model for imputation of scRNA-seq data, helping to improve downstream analysis of single-cell RNA sequencing data and more. 
Without any installation, researchers can download and use the source generation to pre-process scRNA-seq data to accurately and efficiently identify detached values and impute them accurately. 
scNTImpute is fully described and applied in the manuscript "Imputation Methods for Single-Cell RNA-seq Data Using Neural Topic Models". 
This repository includes instructions for use, operating environment and data format requirements

#Folder data
We uploaded a partial dataset (Mouse Pancreatic Islet data) for readers to reproduce in the form of cell-gene (.csv file) containing cell annotation information.
The datasets used in the manuscript have been submitted on the GigaDB database.

#Folder scripts
Readers can realize scNTImpute and its transfer learning through two script files；train_scNTImpute.py and train_Transfer_learning.py.